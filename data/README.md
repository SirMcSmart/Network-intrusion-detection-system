# Dataset

No data is committed to this repository. `src/download_data.py` fetches it into
`data/raw/`, which is git-ignored, and rejects any file whose SHA-256 does not
match `config.yaml`.

## NSL-KDD

- **Publisher:** Canadian Institute for Cybersecurity, University of New
  Brunswick. https://www.unb.ca/cic/datasets/nsl.html
- **Terms:** free for research use, with citation of the paper below. The
  official download sits behind a registration form, so the pipeline fetches
  byte-for-byte copies from two public GitHub mirrors
  (`defcom17/NSL_KDD`, `jmnwong/NSL-KDD-Dataset`). Their row and per-attack
  counts match the figures published with the dataset.
- **Citation:**

  > M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, "A detailed analysis of
  > the KDD CUP 99 data set," *2009 IEEE Symposium on Computational
  > Intelligence for Security and Defense Applications*, 2009, pp. 1-6.

NSL-KDD is a cleaned version of KDD Cup 1999 (itself derived from the 1998
DARPA intrusion detection evaluation): duplicate records removed, and the test
set re-balanced so that easy records do not dominate the score.

| File | Rows | SHA-256 | Role here |
|---|---|---|---|
| `KDDTrain+.txt` | 125,973 | `1b86d2f9…aadf95` | Split 80/20 (stratified) into training and validation |
| `KDDTest+.txt` | 22,544 | `fa46b093…7f84` | Primary test set |
| `KDDTest-21.txt` | 11,850 | `746993ac…531d` | Hard subset of KDDTest+: records that 21 reference classifiers got right fewer than 21 times |

Each row is one connection record: 41 features (3 categorical: protocol,
service, TCP flag), the attack name, and a difficulty score (unused here).

## Attack categories

| Category | Attack types in training | Extra types that appear only in the test sets |
|---|---|---|
| DoS | back, land, neptune, pod, smurf, teardrop | apache2, mailbomb, processtable, udpstorm |
| Probe | ipsweep, nmap, portsweep, satan | mscan, saint |
| R2L (remote to local) | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster | named, sendmail, snmpgetattack, snmpguess, worm, xlock, xsnoop |
| U2R (user to root) | buffer_overflow, loadmodule, perl, rootkit | httptunnel, ps, sqlattack, xterm |

The test-only ("novel") types are the reason NSL-KDD test scores sit well below
validation scores for every model, and they are what the unsupervised vs
supervised comparison is really about. `worm` and `httptunnel` are categorised
inconsistently across the literature; this project follows the most common
convention (R2L and U2R respectively).
