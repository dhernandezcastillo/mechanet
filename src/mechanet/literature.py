"""Literature records cited by the built-in mechanism families and analyses.

Keys are used in ``source_ids`` of nodes, edges and cycles. ``supported_use`` says what each
source supports in this package, so a family is never cited for more than it shows.
"""

LITERATURE = {
    "dionigi2020": {
        "authors": "Dionigi et al.",
        "year": 2020,
        "title": "In-situ structure and catalytic mechanism of NiFe and CoFe layered double "
        "hydroxides during oxygen evolution",
        "journal": "Nature Communications 11, 2522",
        "doi": "10.1038/s41467-020-16237-1",
        "url": "https://doi.org/10.1038/s41467-020-16237-1",
        "material": "Activated gamma-NiFe/CoFe LDH edge models",
        "supported_use": "Surface bridge-OH oxidation and vacancy cycle. Their terminology "
        "separates reacting surface oxygen from lattice oxygen; do not "
        "automatically relabel it bulk LOM.",
    },
    "hao2021": {
        "authors": "Hao et al.",
        "year": 2021,
        "title": "Recognition of Surface Oxygen Intermediates on NiFe Oxyhydroxide Oxygen-Evolving "
        "Catalysts by Homogeneous Oxidation Reactivity",
        "journal": "JACS 143, 1493-1502",
        "doi": "10.1021/jacs.0c11307",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33439638/",
        "material": "NiFe oxyhydroxide",
        "supported_use": "Terminal Fe-O couples to a pre-existing Fe-O-Ni bridge O, forming cyclic O2 "
        "coordination. Detailed refill steps are our explicit hypotheses, not all "
        "resolved in the paper.",
    },
    "song2019": {
        "authors": "Song et al.",
        "year": 2019,
        "title": "An Unconventional Iron Nickel Catalyst for the Oxygen Evolution Reaction",
        "journal": "ACS Central Science 5, 558-568",
        "doi": "10.1021/acscentsci.9b00053",
        "url": "https://doi.org/10.1021/acscentsci.9b00053",
        "material": "FeOOH nanoclusters on NiOOH",
        "supported_use": "Bifunctional O2 formation with a nearby oxygen hydrogen acceptor; proposed "
        "DFT route on a distinct material.",
    },
    "bai2021": {
        "authors": "Bai, Lee and Hu",
        "year": 2021,
        "title": "Spectroscopic and Electrokinetic Evidence for a Bifunctional Mechanism of the Oxygen "
        "Evolution Reaction",
        "journal": "Angewandte Chemie International Edition 60, 3095-3103",
        "doi": "10.1002/anie.202011388",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33089600/",
        "material": "FeOOH-NiOOH compared with NiFe LDH",
        "supported_use": "Supports bifunctional behavior for FeOOH-NiOOH and conventional four-PCET "
        "behavior for the studied NiFe LDH; transfer is not universal.",
    },
    "binninger2022": {
        "authors": "Binninger and Doublet",
        "year": 2022,
        "title": "The Ir-OOOO-Ir transition state and the mechanism of the oxygen evolution reaction on IrO2(110)",
        "journal": "Energy & Environmental Science 15, 2519-2528",
        "doi": "10.1039/D2EE00158F",
        "url": "https://doi.org/10.1039/D2EE00158F",
        "material": "IrO2(110)",
        "supported_use": "Adjacent OO groups release their distal oxygens through an O4 "
        "transition state while anchored oxygens remain. O4 is not established "
        "as a stable minimum.",
    },
    "exner2023": {
        "authors": "Exner",
        "year": 2023,
        "title": "On the mechanistic complexity of oxygen evolution: potential-dependent switching of "
        "the mechanism at the volcano apex",
        "journal": "Materials Horizons 10, 2086",
        "doi": "10.1039/D3MH00047H",
        "url": "https://doi.org/10.1039/D3MH00047H",
        "material": "Generic theoretical OER pathway comparison",
        "supported_use": "Definitions of the mononuclear, bifunctional (BF-I, BF-II), binuclear and "
        "OO/OO-recombination mechanisms and of the potential-dependent descriptor "
        "Gmax(U) (largest free-energy span between intermediates); not an atomistic "
        "proof of any pathway on a given surface.",
    },
    "shi2025": {
        "authors": "Shi et al.",
        "year": 2025,
        "title": "Lattice O-O ligands in Fe-incorporated hydroxides enhance water oxidation electrocatalysis",
        "journal": "Nature Chemistry 17, 1607-1614",
        "doi": "10.1038/s41557-025-01898-6",
        "url": "https://pubmed.ncbi.nlm.nih.gov/40826234/",
        "material": "Reconstructed NiFe superoxo-hydroxide",
        "supported_use": "Stable support O-O ligands and Fe AEM; explicit local water/proton "
        "rearrangement. A displayed two-PCET transformation needs resolution before "
        "calling it elementary.",
    },
    "ha2024": {
        "authors": "Ha et al.",
        "year": 2024,
        "title": "Fe-Doped Ni-Based Catalysts Surpass Ir-Baselines for Oxygen Evolution Due to Optimal "
        "Charge-Transfer Characteristics",
        "journal": "ACS Catalysis 14, 17347-17359",
        "doi": "10.1021/acscatal.4c04489",
        "url": "https://doi.org/10.1021/acscatal.4c04489",
        "material": "Fe-substituted rock-salt NiO(100) (not an oxyhydroxide)",
        "supported_use": "Related-material precedent for OH/O/H2O isomers and O+O or OOH+OH coupling. "
        "The 2025 correction concerns acknowledgments.",
    },
    "grimaud2017": {
        "authors": "Grimaud et al.",
        "year": 2017,
        "title": "Activating lattice oxygen redox reactions in metal oxides to catalyse oxygen evolution",
        "journal": "Nature Chemistry 9, 457-465",
        "doi": "10.1038/nchem.2695",
        "url": "https://doi.org/10.1038/nchem.2695",
        "material": "Perovskite oxides",
        "supported_use": "Lattice oxygen participation and schematic "
        "two-framework-oxygen/two-vacancy pathway; not proof for other oxides.",
    },
    "lee2019": {
        "authors": "Lee et al.",
        "year": 2019,
        "title": "Oxygen Isotope Labeling Experiments Reveal Different Reaction Sites for the Oxygen "
        "Evolution Reaction on Nickel and Nickel Iron Oxides",
        "journal": "Angewandte Chemie International Edition 58, 10295-10299",
        "doi": "10.1002/anie.201903200",
        "url": "https://doi.org/10.1002/anie.201903200",
        "material": "Ni, NiCo, NiFe and NiCoFe hydroxides",
        "supported_use": "Control evidence: some O-O species can be spectators, and lattice "
        "participation differs between materials.",
    },
    "lee2023": {
        "authors": "Lee et al.",
        "year": 2023,
        "title": "Operando identification of a side-on nickel superoxide intermediate and the mechanism "
        "of oxygen evolution on nickel oxyhydroxide",
        "journal": "Chem Catalysis 3, 100475",
        "doi": "10.1016/j.checat.2022.11.014",
        "url": "https://doi.org/10.1016/j.checat.2022.11.014",
        "material": "NiOOH without Fe",
        "supported_use": "Side-on/bridging O2-like geometry and a role beyond immediate O2 release; "
        "supports exploring binding variants, not assigning formal charge from "
        "geometry.",
    },
    "halck2014": {
        "authors": "Halck, Petrykin, Krtil and Rossmeisl",
        "year": 2014,
        "title": "Beyond the volcano limitations in electrocatalysis - oxygen evolution reaction",
        "journal": "Physical Chemistry Chemical Physics 16, 13682-13688",
        "doi": "10.1039/C4CP00571F",
        "url": "https://doi.org/10.1039/C4CP00571F",
        "material": "Model oxide surfaces with a neighbouring proton-acceptor site",
        "supported_use": "Original proposal of the bifunctional OER mechanism, in which a proton from "
        "the water-attack step is transferred to a neighbouring acceptor site.",
    },
    "norskov2004": {
        "authors": "Norskov, Rossmeisl, Logadottir, Lindqvist, Kitchin, Bligaard and Jonsson",
        "year": 2004,
        "title": "Origin of the overpotential for oxygen reduction at a fuel-cell cathode",
        "journal": "Journal of Physical Chemistry B 108, 17886-17892",
        "doi": "10.1021/jp047349j",
        "url": "https://doi.org/10.1021/jp047349j",
        "material": "Computational hydrogen electrode",
        "supported_use": "CHE bookkeeping; gas-phase H2O taken at 0.035 bar (equilibrium with "
        "liquid water near 300 K) as the liquid-water reference.",
    },
    "rossmeisl2007": {
        "authors": "Rossmeisl, Qu, Zhu, Kroes and Norskov",
        "year": 2007,
        "title": "Electrolysis of water on oxide surfaces",
        "journal": "Journal of Electroanalytical Chemistry 607, 83-89",
        "doi": "10.1016/j.jelechem.2006.11.008",
        "url": "https://doi.org/10.1016/j.jelechem.2006.11.008",
        "material": "Rutile oxides (RuO2, IrO2, TiO2)",
        "supported_use": "Mononuclear (adsorbate evolution) OER on coordinatively unsaturated "
        "metal sites of rutile (110) surfaces; CHE thermodynamic analysis.",
    },
    "rossmeisl2005": {
        "authors": "Rossmeisl, Logadottir and Norskov",
        "year": 2005,
        "title": "Electrolysis of water on (oxidized) metal surfaces",
        "journal": "Chemical Physics 319, 178-184",
        "material": "Pt(111), Au(111) and other (oxidized) metal surfaces",
        "supported_use": "Adsorbate evolution OER on metal top sites with the computational hydrogen "
        "electrode; the reference analysis for the Pt(111) example.",
    },
    "man2011": {
        "authors": "Man et al.",
        "year": 2011,
        "title": "Universality in Oxygen Evolution Electrocatalysis on Oxide Surfaces",
        "journal": "ChemCatChem 3, 1159-1165",
        "doi": "10.1002/cctc.201000397",
        "url": "https://doi.org/10.1002/cctc.201000397",
        "material": "Rutile, perovskite, spinel, rock-salt and bixbyite oxides",
        "supported_use": "Mononuclear OER on oxide surfaces, OOH/OH scaling and the thermodynamic "
        "overpotential volcano.",
    },
    "hansen2008": {
        "authors": "Hansen, Rossmeisl and Norskov",
        "year": 2008,
        "title": "Surface Pourbaix diagrams and oxygen reduction activity of Pt, Ag and Ni(111) "
        "surfaces studied by DFT",
        "journal": "Physical Chemistry Chemical Physics 10, 3722-3730",
        "doi": "10.1039/B803956A",
        "url": "https://doi.org/10.1039/B803956A",
        "material": "Pt, Ag and Ni(111)",
        "supported_use": "Surface Pourbaix diagrams: the most stable surface coverage as a function "
        "of potential and pH from CHE formation free energies.",
    },
    "hansen2010": {
        "authors": "Hansen, Man, Studt, Abild-Pedersen, Bligaard and Rossmeisl",
        "year": 2010,
        "title": "Electrochemical chlorine evolution at rutile oxide (110) surfaces",
        "journal": "Physical Chemistry Chemical Physics 12, 283-290",
        "doi": "10.1039/B917459A",
        "url": "https://doi.org/10.1039/B917459A",
        "material": "Rutile oxide (110) surfaces (IrO2, RuO2, PtO2, TiO2)",
        "supported_use": "CER intermediates (Cl and OCl on cus sites) chosen from the surface "
        "Pourbaix diagram; Volmer-Heyrovsky-type CHE analysis referenced to the "
        "Cl2/Cl- couple.",
    },
    "exner2020": {
        "authors": "Exner",
        "year": 2020,
        "title": "A Universal Descriptor for the Screening of Electrode Materials for "
        "Multiple-Electron Processes: Beyond the Thermodynamic Overpotential",
        "journal": "ACS Catalysis 10, 12607-12617",
        "doi": "10.1021/acscatal.0c03865",
        "url": "https://doi.org/10.1021/acscatal.0c03865",
        "material": "Generic multi-electron electrocatalysis",
        "supported_use": "Definition of the potential-dependent descriptor Gmax(eta) based on the "
        "free-energy span model.",
    },
    "razzaq2023": {
        "authors": "Razzaq and Exner",
        "year": 2023,
        "title": "Materials Screening by the Descriptor Gmax(eta): The Free-Energy Span Model in Electrocatalysis",
        "journal": "ACS Catalysis 13, 1740-1758",
        "doi": "10.1021/acscatal.2c03997",
        "url": "https://doi.org/10.1021/acscatal.2c03997",
        "material": "Review of Gmax(eta) applications",
        "supported_use": "Use of Gmax(eta) across OER, ORR and selectivity problems.",
    },
    "govindrajan2020": {
        "authors": "Govind Rajan and Carter",
        "year": 2020,
        "title": "Discovering Competing Electrocatalytic Mechanisms and their Overpotentials: "
        "Automated Enumeration of Oxygen Evolution Pathways",
        "journal": "Journal of Physical Chemistry C",
        "doi": "10.1021/acs.jpcc.0c08120",
        "url": "https://doi.org/10.1021/acs.jpcc.0c08120",
        "material": "NiOOH and Fe-doped NiOOH surfaces",
        "supported_use": "Automated enumeration of OER pathways over a library of surface "
        "species; motivates graph-based pathway discovery.",
    },
}
