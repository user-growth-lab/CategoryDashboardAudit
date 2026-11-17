# Third-Party Notices

The root MIT `LICENSE` applies only to original code and documentation for which
the authors hold the necessary rights. It does not grant rights in the
datasets, metadata, dependencies, fonts, conference materials, or other
third-party assets described below.

## KuaiRand-Pure

The real-data stages use KuaiRand-Pure random-insertion and standard-serving
interaction logs. The dataset is not redistributed in this public
repository. Readers who elect to rerun those stages must obtain it from the
official sources:

- project repository: <https://github.com/chongminggao/KuaiRand>;
- versioned archive record: <https://zenodo.org/records/10439422>;
- archive name: `KuaiRand-Pure.tar.gz`;
- published archive MD5: `0820331067a3784d9691136f772b35a7`.

The official project repository displays CC BY-SA 4.0, while the checked
versioned archive record does not consistently expose an equivalent file-level
license statement.  This release therefore makes no independent licensing
claim and redistributes no KuaiRand rows or row-level derivatives.

## KuaiRand category supplement

First-level category labels come from KuaiRand's official predicted-category
supplement, not from a KuaiRec table:

- record: <https://zenodo.org/records/18159199>;
- file: `kuairand_video_categories.csv`;
- published MD5: `8a4c772d3d8f8cf65fa1e1eb896f4177`;
- join key: `final_video_id`.

The labels are predicted taxonomy assignments with associated probabilities,
not manually verified category ground truth.  Neither the complete supplement
nor the derived Pure-ID subset is redistributed here.

## Software dependencies and PDF assets

Packages listed in `requirements.txt` retain their own licenses and attribution
requirements.  Fonts and other third-party assets embedded in the prebuilt
technical-appendix PDF retain their respective licenses.  No dependency source
tree, model weight, embedding, ACM template source, or upstream repository
checkout is redistributed under the root MIT License.

The focused P7--P14 public repository excludes the earlier KuaiRec caption
pipeline and its external text encoder entirely.
