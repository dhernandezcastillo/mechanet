"""Reactions: reservoir species, net stoichiometry and the rules that balance each step.

Every edge of a network exchanges atoms and electrons with reservoirs. An edge carries a
*reservoir vector* ``nu``: the amount of each reservoir species it produces (negative when
consumed), for example ``{"H2O": -1, "H+": 1}`` for ``*O + H2O -> *OOH + H+ + e-``.
Electrons follow from charge balance: ``electrons = sum(nu[s] * charge[s])``, positive
when released to the electrode (anodic).

The surface composition change of an edge ``i -> j`` satisfies, for every element E,
``n_E(j) - n_E(i) = -sum_s nu[s] * composition[s][E]``.

A reaction declares

* ``net``: the reservoir vector of one turnover (OER: 2 H2O -> O2 + 4 H+ + 4 e-),
* ``release``: the *turnover marker*, a species of ``net`` with coefficient +1 (a product
  released once per turnover: O2 for OER, Cl2 for CER, H2 for HER) or -1 (a reactant
  consumed once per turnover: O2 for ORR); a cycle contains exactly one step that moves it
  in the reaction's direction,
* ``balance_species``: the species that absorbs each element when a step is written
  with only some coefficients given (OER: H goes to H+),
* ``potential_scale``: the scale on which potentials are reported (RHE for OER, SHE for CER),
* ``elementary``: reservoir vectors that count as one elementary step, used by
  ``mechanet pathways`` to propose new edges.

New reactions (ORR, HER, H2O2, ...) are plugins: subclass :class:`Reaction` and register
it in the ``mechanet.reactions`` entry-point group. :func:`validate_reaction` checks the
declaration when the reaction is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional


@dataclass(frozen=True)
class Species:
    """A reservoir species.

    ``references``: the molecules of the references file its chemical potential needs.
    ``chemical_potential``: ``f(reservoirs, conditions) -> eV`` at U_SHE = 0 (electrons are
    added separately). Built-in species use the rules of :mod:`mechanet.thermo`; a plugin
    species without a rule is treated as a neutral molecule: mu = E0 + G_corr of the
    reference molecule of the same name, plus kT ln(activity).
    """

    name: str
    composition: tuple  # ((element, count), ...)
    charge: int
    phase: str
    references: tuple = ()
    chemical_potential: Optional[Callable] = None

    def count(self, element):
        return dict(self.composition).get(element, 0)


SPECIES = {
    "H2O": Species("H2O", (("H", 2), ("O", 1)), 0, "liquid", ("H2O",)),
    "H+": Species("H+", (("H", 1),), 1, "aqueous", ("H2",)),
    "O2": Species("O2", (("O", 2),), 0, "gas", ("H2", "H2O")),
    "Cl-": Species("Cl-", (("Cl", 1),), -1, "aqueous", ("Cl2",)),
    "Cl2": Species("Cl2", (("Cl", 2),), 0, "gas", ("Cl2",)),
}
BUILTIN_SPECIES = frozenset(SPECIES)


def register_species(species):
    """Make a reservoir species available to every reaction (plugins call registry.add_species)."""
    known = SPECIES.get(species.name)
    if known is not None and known != species:
        raise ValueError(f"Reservoir species {species.name!r} is already registered with different data")
    if species.charge and species.chemical_potential is None:
        raise ValueError(f"Charged species {species.name!r} needs a chemical_potential rule")
    SPECIES[species.name] = species


OER_TRIO = ("H2O", "H+", "O2")


def electrons(vector):
    """Electrons released to the electrode by a reservoir vector (charge balance)."""
    return int(round(sum(float(nu) * SPECIES[name].charge for name, nu in vector.items())))


def clean(vector):
    """Drop zero entries and normalise integral floats to int."""
    out = {}
    for name, nu in vector.items():
        if name not in SPECIES:
            raise ValueError(f"Unknown reservoir species {name!r}; known: {sorted(SPECIES)}")
        value = float(nu)
        if abs(value) > 1e-12:
            out[name] = int(round(value)) if abs(value - round(value)) < 1e-9 else value
    return dict(sorted(out.items()))


def element_change(vector):
    """Surface composition change implied by a reservoir vector (element -> count)."""
    change = {}
    for name, nu in vector.items():
        for element, count in SPECIES[name].composition:
            change[element] = change.get(element, 0) - nu * count
    return {k: v for k, v in change.items() if abs(v) > 1e-12}


def add(*vectors):
    total = {}
    for vector in vectors:
        for name, nu in vector.items():
            total[name] = total.get(name, 0) + nu
    return clean(total)


def oer_trio(vector):
    """(water, pairs, oxygen) of an OER-only vector, else None (edge IDs and OER edge tables)."""
    if set(vector) - set(OER_TRIO):
        return None
    return (-vector.get("H2O", 0), vector.get("H+", 0), vector.get("O2", 0))


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction(float(value)).limit_denominator(10**6)


def _row_reduce(columns, target=None):
    """Exact Gauss-Jordan elimination of the element-balance system.

    ``columns``: one composition per species (element -> count); ``target``: element -> count
    (zero if None). Pivots are taken in column order, so earlier species are preferred.
    Returns (solution or None if inconsistent, rank); free (later) species are set to 0.
    """
    target = target or {}
    elements = sorted({e for column in columns for e in column} | set(target))
    n = len(columns)
    rows = [[_fraction(column.get(e, 0)) for column in columns] + [_fraction(target.get(e, 0))] for e in elements]
    pivots, r = [], 0
    for c in range(n):
        pivot = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        lead = rows[r][c]
        rows[r] = [x / lead for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
        if r == len(rows):
            break
    if any(all(x == 0 for x in row[:n]) and row[n] != 0 for row in rows):
        return None, len(pivots)
    solution = [Fraction(0)] * n
    for i, c in enumerate(pivots):
        solution[c] = rows[i][n]
    return solution, len(pivots)


def validate_reaction(reaction):
    """Check a reaction declaration; raises ValueError with what to change.

    ``net`` must use registered species, balance every element and move electrons; the
    turnover marker ``release`` must appear in ``net`` with coefficient +1 (released once
    per turnover) or -1 (consumed once per turnover).
    """
    name = getattr(reaction, "name", type(reaction).__name__)
    net = clean(reaction.net)
    if not net:
        raise ValueError(f"Reaction {name!r}: net is empty")
    leftover = element_change(net)
    if leftover:
        raise ValueError(f"Reaction {name!r}: net {net} does not balance; it changes the surface by {leftover}")
    if electrons(net) == 0:
        raise ValueError(f"Reaction {name!r}: net {net} transfers no electrons; the descriptors need an electrode")
    marker = reaction.release
    if marker not in net or abs(net[marker]) != 1:
        candidates = sorted(s for s, nu in net.items() if abs(nu) == 1)
        raise ValueError(
            f"Reaction {name!r}: release = {marker!r} must be a species of net with coefficient +1 (a product "
            f"released once per turnover) or -1 (a reactant consumed once per turnover); net {net} allows "
            f"{candidates or 'none'}. For example ORR, O2 + 4 H+ + 4 e- -> 2 H2O, uses release = 'O2' (consumed)."
        )
    for vector in reaction.elementary:
        clean(vector)
    for species in reaction.formation_species:
        if species not in SPECIES:
            raise ValueError(f"Reaction {name!r}: unknown formation species {species!r}")
    return True


class Reaction:
    name = "reaction"
    title = "Abstract reaction"
    net: dict = {}
    release = None
    balance_species: dict = {}
    formation_species: tuple = ()
    potential_scale = "SHE"
    default_potentials: tuple = ()
    required_references: tuple = ()
    elementary: tuple = ()

    @property
    def electrons(self):
        return electrons(self.net)

    @property
    def species(self):
        return tuple(sorted(set(self.net) | set(self.balance_species.values()) | set(self.formation_species)))

    @property
    def marker_sign(self):
        """+1 when the turnover marker (``release``) is produced, -1 when it is consumed."""
        return 1 if float(self.net.get(self.release, 0)) > 0 else -1

    def advances(self, vector):
        """True when a step moves the turnover marker in the reaction's direction."""
        return float(vector.get(self.release, 0)) * self.marker_sign > 0

    def opposes(self, vector):
        """True when a step moves the turnover marker against the reaction's direction."""
        return float(vector.get(self.release, 0)) * self.marker_sign < 0

    @property
    def closed_walks_are_turnovers(self):
        """True when every element-balanced combination of the reaction's species is a multiple
        of ``net``: a closed walk over its species that moves the marker once is then exactly
        one turnover (OER, CER, HER, ORR, NRR). Otherwise cycles must be checked one by one."""
        names = self.species
        _, rank = _row_reduce([dict(SPECIES[s].composition) for s in names])
        return len(names) - rank == 1

    def balance(self, delta, given=None):
        """Complete a step's reservoir vector from the surface change ``delta`` (element -> count).

        ``given`` fixes some coefficients (for OER the families give water and O2); the
        balance species of the reaction take up what is left, solved exactly (in any order
        of ``balance_species``), and every element must then balance.
        """
        vector = dict(given or {})
        residual = {element: -_fraction(count) for element, count in delta.items()}
        for name, nu in vector.items():
            for element, count in SPECIES[name].composition:
                residual[element] = residual.get(element, 0) - _fraction(nu) * count
        unknown = [name for name in dict.fromkeys(self.balance_species.values()) if name not in vector]
        solution, _ = _row_reduce([dict(SPECIES[name].composition) for name in unknown], residual)
        if solution is None:
            leftover = {k: float(v) for k, v in residual.items() if v != 0}
            raise ValueError(
                f"{self.name}: step does not balance; unaccounted atoms {leftover} "
                f"(surface change {delta}, given {given})"
            )
        vector.update({name: float(x) for name, x in zip(unknown, solution) if x != 0})
        return clean(vector)

    def formation_vector(self, delta):
        """Reservoir vector that forms a node from the reference surface (Pourbaix / resting states).

        Solves the element balance exactly (rational arithmetic) for the formation species, in
        any order; when the species admit several solutions (e.g. H2O, H+ and OH-, related by
        water autoionization), the earlier species in ``formation_species`` are used.
        """
        names = tuple(self.formation_species)
        target = {element: -_fraction(count) for element, count in delta.items() if float(count) != 0}
        solution, _ = _row_reduce([dict(SPECIES[name].composition) for name in names], target)
        if solution is None:
            raise ValueError(
                f"{self.name}: cannot form a node with surface change {delta} from the formation species "
                f"{names}; add a species that carries the missing element to formation_species"
            )
        return clean({name: float(x) for name, x in zip(names, solution) if x != 0})

    def is_turnover(self, vector):
        return clean(vector) == clean(self.net)

    def describe(self):
        return {
            "name": self.name,
            "title": self.title,
            "net": clean(self.net),
            "electrons": self.electrons,
            "release": self.release,
            "potential_scale": self.potential_scale,
            "required_references": list(self.required_references),
        }


class OER(Reaction):
    """Oxygen evolution, 2 H2O -> O2 + 4 (H+ + e-), with the computational hydrogen electrode."""

    name = "oer"
    title = "Oxygen evolution reaction"
    net = {"H2O": -2, "H+": 4, "O2": 1}
    release = "O2"
    balance_species = {"H": "H+"}
    formation_species = ("H2O", "H+")
    potential_scale = "RHE"
    default_potentials = (1.23, 1.43, 1.53, 1.63)
    required_references = ("H2", "H2O")
    elementary = (
        {"H+": 1},  # proton-coupled electron transfer
        {"H2O": -1, "H+": 1},  # water attack or vacancy refill with one PCET
        {"H2O": -1},  # water adsorption, or water attack with H to a neighbour
        {"H2O": 1},  # water desorption
        {},  # internal H transfer, O-O coupling
        {"O2": 1, "H+": 1},  # O2 release with one PCET
        {"O2": 1},  # chemical O2 release
    )


class CER(Reaction):
    """Chlorine evolution, 2 Cl- -> Cl2 + 2 e-, referenced to the Cl2/Cl- couple.

    Water and protons stay available as reservoirs so that oxygen-covered sites (*O formed
    from water) and OCl intermediates can be written; the turnover itself is
    ``{"Cl-": -2, "Cl2": 1}``. In alkaline solution chloride oxidation gives hypochlorite,
    which needs its own reservoir and is not covered by this plugin.
    """

    name = "cer"
    title = "Chlorine evolution reaction"
    net = {"Cl-": -2, "Cl2": 1}
    release = "Cl2"
    balance_species = {"H": "H+", "Cl": "Cl-"}
    formation_species = ("H2O", "H+", "Cl-")
    potential_scale = "SHE"
    default_potentials = (1.36, 1.46, 1.56)
    required_references = ("H2", "H2O", "Cl2")
    elementary = OER.elementary + (
        {"Cl-": -1},  # Volmer: * + Cl- -> *Cl + e-  (also *O + Cl- -> *OCl + e-)
        {"Cl-": -1, "Cl2": 1},  # Heyrovsky: *Cl + Cl- -> * + Cl2 + e-
        {"Cl2": 1},  # Tafel: 2 *Cl -> 2 * + Cl2
    )


BUILTIN_REACTIONS = {"oer": OER, "cer": CER}
