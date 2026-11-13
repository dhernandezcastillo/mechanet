# Model files on a cluster

The ML potentials of [`mechanet prescreen`](prescreening.md) each need one model file:
`mace-mh1` loads MACE-MH-1, `uma` loads UMA `uma-s-1p2p1` and `sevennet` loads SevenNet-Omni.
On a cluster, the safest way is to get each file once, keep it in one folder that every job can
read, and let the jobs read it from there. The jobs then never need the internet, and UMA's
Hugging Face login happens once, in your web browser, never on the cluster.

## Where mechanet looks

A calculator takes its model file from the first of:

1. the path you give, `--option model=/path/to/file`;
2. the **models folder**: the folder that `MECHANET_MODELS` names, else
   `~/.local/share/mechanet/models` (`$XDG_DATA_HOME/mechanet/models` when that variable is
   set). The file sits there under its official name (table below), directly or in a folder
   up to two levels down;
3. the package's own copy: mace-torch, fairchem and sevenn download the model on the first run
   and keep it. This needs the internet, and for UMA a Hugging Face login with access to the
   weights.

A file found in 1 or 2 is read from disk and nothing is downloaded. Every job writes to its log
which model file it loaded and where it found it.

Prefer the models folder to `--option model=PATH`. Results record the model by its name, its
official file name and its SHA-256, not by where it sits: they stay valid when the folder moves,
and the copy in the models folder counts as the same model as the package's own download of the
same file.

| Calculator | File | Size | Download | Licence | Account |
|---|---|---|---|---|---|
| `mace-mh1` | `mace-mh-1.model` | 59 MB | [GitHub release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model) | [Academic Software License](https://github.com/gabor1/ASL): academic, non-commercial use only | none |
| `uma` | `uma-s-1p2p1.pt` | 2.3 GB | [Hugging Face](https://huggingface.co/facebook/UMA/resolve/main/checkpoints/uma-s-1p2p1.pt) | FAIR Chemistry License, on the [model page](https://huggingface.co/facebook/UMA) | Hugging Face, with access granted to facebook/UMA |
| `sevennet` | `checkpoint_sevennet_omni.pth` | 103 MB | [GitHub release](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth) | [MIT](https://github.com/MDIL-SNU/SevenNet/blob/main/LICENSE) | none |

mechanet never ships or mirrors these files: each comes from its official page, under its own
licence. MACE-MH-1's licence allows academic, non-commercial use only; commercial use needs a
licence from its authors. UMA's licence asks you to acknowledge UMA in publications.

## Three ways to fill the models folder

### A. Download in your browser, copy to the cluster

No login and no internet are needed on the cluster: the simplest way for most people.

1. On your own computer, download the files from the links in the table. For UMA, log in to
   Hugging Face first and ask for access on the [model page](https://huggingface.co/facebook/UMA);
   requests are reviewed, so it is not instant. Once granted, the download link works in that
   browser.
2. On the cluster, create the models folder:

   ```bash
   mkdir -p ~/.local/share/mechanet/models
   ```

3. Copy the files into it the way you copy structures: WinSCP, MobaXterm, the cluster's web
   portal, or from your computer:

   ```bash
   scp mace-mh-1.model uma-s-1p2p1.pt you@cluster:~/.local/share/mechanet/models/
   ```

4. Check on the login node, [below](#check-before-you-submit).

### B. Download on the login node

When the login node has internet. MACE-MH-1 and SevenNet-Omni need no account; download them
straight into the models folder:

```bash
cd ~/.local/share/mechanet/models
```

```bash
curl -LO https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model
```

```bash
curl -LO https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth
```

UMA needs a Hugging Face login on the cluster, once (`hf` comes with fairchem, in the UMA
environment):

```bash
hf auth login
```

```bash
hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir ~/.local/share/mechanet/models/UMA
```

The file lands in `UMA/checkpoints/`, two levels down, where mechanet finds it.

### C. One folder for a group

One person fills a folder that the whole group can read, for instance on project storage, and
everyone names it in their `~/.bashrc`:

```bash
export MECHANET_MODELS=/project/mygroup/mechanet-models
```

One download serves everyone, and everyone uses the identical files, which keeps results
comparable across the group. Ask your cluster's support first: some keep such copies already.

UMA's licence (the FAIR Chemistry License) lets you pass copies on only together with the
licence text: put UMA's `LICENSE` file, from the model page, next to `uma-s-1p2p1.pt`. Uses
must follow Meta's acceptable-use policy, and publications must acknowledge UMA. Have everyone
who uses UMA accept its terms on Hugging Face as well, so each has agreed to them. Read the
licence on the model page before sharing; mechanet's summary of it is no substitute.

## Check before you submit

```bash
mechanet models
```

This reports, for each calculator:

- which file a job would load and where it is;
- whether the file is complete: a download cut short, or an error page saved under the model's
  name, shows here;
- whether its package is installed in this environment, and recent enough;
- whether a job can run without internet, and where to get what is missing.

It downloads nothing. Each calculator lives in its own environment (mace-torch needs an e3nn
version the others reject), so check each in its own: `mechanet models uma` in the UMA
environment. A shortened example:

```text
Models folder: /home/ana/.local/share/mechanet/models (the default; MECHANET_MODELS sets another)

mace-mh1: MACE-MH-1, file mace-mh-1.model
  file     /home/ana/.local/share/mechanet/models/mace-mh-1.model (models folder)
           59,208,139 bytes, complete
  package  mace-torch 0.3.16
  licence  Academic Software License (ASL): academic, non-commercial use only
  ready    yes, without internet
```

`--load` also loads each model found on this computer, as a job would and without the internet,
and computes one small structure (a Pt(111) slab with O). It also checks every byte of the file
and prints the SHA-256 that results record. A wrong package version or a missing GPU driver then
shows in a minute, not after hundreds of jobs. To test the GPUs, run it in a short interactive
job on a GPU node, with `--device cuda`.

## In a job script

`mechanet models CALCULATOR` exits with status 1 when that calculator is not ready, so a job can
stop before it relaxes anything:

```bash
source ~/venvs/mechanet-uma/bin/activate
mechanet models uma || exit 1
mechanet prescreen --calculator uma --option device=cuda
```

Add `export MECHANET_MODELS=...` before them when the models folder is not the default one.

## The packages' own copies

A model that its package downloaded on an earlier run works too. Jobs find it only when they
see the same cache folder (`~/.cache/mace`, `~/.cache/fairchem` or `FAIRCHEM_CACHE_DIR`,
`~/.cache/sevennet`). fairchem also first tries to reach Hugging Face unless the job sets
`HF_HUB_OFFLINE=1`. `mechanet models` reports such copies and says so. The models folder needs
neither.

## UMA's two reference files

fairchem's own download also fetches two small files from the model page,
`references/iso_atom_elem_refs.yaml` and `references/form_elem_refs.yaml`. fairchem uses them
only for an isolated single atom (one atom alone in a box, without periodic boundaries) and in
its formation-energy wrapper. The energies and forces of slabs and molecules come from the model
file, which carries its own element references. mechanet uses neither, so the model file alone
gives the same results (read in fairchem-core 2.24's source).

## Avoid

- **Letting every job download.** Many jobs starting together each fetch the model (2.3 GB for
  UMA): slow, can hit download limits, and fails at random. Nodes without internet fail
  outright.
- **Copying files into the packages' hidden caches by hand.** Each package arranges its
  downloads its own way:
  - fairchem uses Hugging Face's folder layout;
  - mace-torch saves `mace-mh-1.model` as `macemh1model`;
  - sevenn uses folders of its own.

  A file dropped there is not found. Use the models folder.
- **Putting your Hugging Face token in job scripts or shared folders.**
- **Using copies of UMA from anywhere but its official page.**
