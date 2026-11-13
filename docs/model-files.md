# Model files on a cluster

The ML potentials of [`mechanet prescreen`](prescreening.md) each need one model file:
`mace-mh1` loads MACE-MH-1, `uma` loads UMA `uma-s-1p2p1` and `sevennet` loads SevenNet-Omni.
On a cluster, the safest way is to get each file once, keep it in one folder that every job can
read, and let the jobs read it from there. The jobs then never need the internet, nor, for UMA,
a Hugging Face token.

## Where mechanet looks

A calculator takes its model file from the first of:

1. the path you give, `--option model=/path/to/file`;
2. the **models folder**: the folder that `MECHANET_MODELS` names, else
   `~/.local/share/mechanet/models` (`$XDG_DATA_HOME/mechanet/models` when that variable is
   set). The file sits there under its official name (table below), directly or in a folder
   up to two levels down;
3. the package's own copy: mace-torch, fairchem and sevenn download the model on the first run
   and keep it. This needs the internet, and for UMA a Hugging Face token with access to the
   weights ([UMA with a Hugging Face token](#uma-with-a-hugging-face-token)).

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

When the login node has internet, `mechanet models --download` puts the files in the models
folder, from the links in the table. Name the models you want: each comes under its own
licence, which is shown as it downloads. MACE-MH-1 and SevenNet-Omni need no account:

```bash
mechanet models mace-mh1 sevennet --download
```

UMA needs your Hugging Face token, set up once as in
[UMA with a Hugging Face token](#uma-with-a-hugging-face-token). Then, in the UMA environment:

```bash
mechanet models uma --download
```

UMA lands in `UMA/checkpoints/` inside the models folder, with UMA's `LICENSE` and
`USE_POLICY.md` next to it, the layout `hf download facebook/UMA checkpoints/uma-s-1p2p1.pt
--local-dir UMA` gives too. Jobs then read it there, with neither internet nor token. A
download cut short is never mistaken for the model: run the command again. A file already in
the folder is kept, and the report that follows says whether each model is ready.

### C. One folder for a group

One person fills a folder that the whole group can read, for instance on project storage, and
everyone names it in their `~/.bashrc`:

```bash
export MECHANET_MODELS=/project/mygroup/mechanet-models
```

One download serves everyone, and everyone uses the identical files, which keeps results
comparable across the group. Ask your cluster's support first: some keep such copies already.

UMA's licence (the FAIR Chemistry License) lets you pass copies on only together with the
licence text. `mechanet models uma --download` saves UMA's `LICENSE` and `USE_POLICY.md` next
to `uma-s-1p2p1.pt`; when you copy the file in by hand, put the `LICENSE` file from the model
page next to it. Uses must follow Meta's acceptable-use policy, and publications must
acknowledge UMA. Have everyone who uses UMA accept its terms on Hugging Face as well, so each
has agreed to them. Read the licence on the model page before sharing; mechanet's summary of it
is no substitute.

## UMA with a Hugging Face token

This is the way UMA's own documentation describes. Set it up once:

1. Log in to Hugging Face and ask for access on the
   [model page](https://huggingface.co/facebook/UMA). Requests are reviewed: wait for the
   e-mail that grants it.
2. Create a token at <https://huggingface.co/settings/tokens>. UMA's documentation asks for the
   permission "Read access to contents of all public gated repos you can access": a token of
   type **Read** has it; for a **Fine-grained** token, tick it.
3. Give the token to the computer that downloads (on a cluster, the login node). `hf` comes with
   fairchem, so run this in the UMA environment:

   ```bash
   hf auth login
   ```

   and paste the token when asked (`huggingface-cli login` in older versions). Recent versions
   first ask how to log in: choose "Paste an access token". If asked whether to add the token
   as a git credential, answer no. The token is saved in your home folder
   (`~/.cache/huggingface/token`), where later downloads find it.

   Or set the `HF_TOKEN` variable for this session only. This line reads it without showing it
   or keeping it in the shell's history (paste the token, then press Enter):

   ```bash
   read -rs HF_TOKEN && export HF_TOKEN
   ```

Then either:

- **download UMA into the models folder** (way B): `mechanet models uma --download`. Best on a
  cluster: jobs then need neither the internet nor the token;
- or **let fairchem download it on the first run**, as UMA's documentation does: when the
  computer that runs `mechanet prescreen --calculator uma` has internet, nothing more is needed,
  and the model goes to fairchem's cache (`~/.cache/fairchem`, or the folder in
  `FAIRCHEM_CACHE_DIR`). Fine on your own computer; on a cluster, many jobs starting together
  would each download 2.3 GB, and nodes without internet cannot.

`mechanet models uma` says whether a token is found, and where: in `HF_TOKEN`, or in the file
that `hf auth login` saved. Only Hugging Face can tell whether the token gives access to UMA:
`--download` finds out.

**Keep the token to yourself.** It works like a password to your Hugging Face account:

- never put it in job scripts, shared folders, notebooks you share, or mechanet's options;
- a token of type Read can only read: it cannot change anything on your account;
- if it may have leaked, or you no longer need it, delete it at
  <https://huggingface.co/settings/tokens>; `hf auth logout` removes it from the computer.

mechanet never shows, stores or records the token: huggingface_hub reads it, and results record
only the model file's name and SHA-256.

**When Hugging Face refuses**, mechanet says which case it is:

- no token found here (401): steps 1 to 3;
- a token that Hugging Face did not accept (401), mistyped, expired or deleted: create a new
  one and log in again with it (recent versions need `hf auth login --force` to replace a saved
  token);
- no access yet (403): the request of step 1 is still waiting, or a fine-grained token lacks
  the permission of step 2;
- "HF_HUB_OFFLINE is set": run `unset HF_HUB_OFFLINE` for the download; jobs may keep it.

## Check before you submit

```bash
mechanet models
```

This reports, for each calculator:

- which file a job would load and where it is;
- whether the file is complete: a download cut short, or an error page saved under the model's
  name, shows here;
- whether its package is installed in this environment, and recent enough;
- whether a job can run without internet, and where to get what is missing;
- for UMA, when its file is missing, whether a Hugging Face token is found, and where (never
  the token itself).

It downloads nothing unless you add `--download`. Each calculator lives in its own environment
(mace-torch needs an e3nn version the others reject), so check each in its own:
`mechanet models uma` in the UMA environment. A shortened example:

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
- **Putting your Hugging Face token in job scripts, shared folders or mechanet's options.**
  huggingface_hub finds it where `hf auth login` saved it.
- **Using copies of UMA from anywhere but its official page.**
