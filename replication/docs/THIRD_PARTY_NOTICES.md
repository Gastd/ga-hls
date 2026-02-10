# Third-party notices (ga-hls)

This repository is MIT-licensed (see `LICENSE`). The Docker image used for replication downloads third-party components at build time.

## Weka (weka-stable)

- Component: `weka-stable` (Java ML workbench, used to run J48)
- Version used by the Dockerfile: **3.8.6**
- Source: Maven Central coordinates `nz.ac.waikato.cms.weka:weka-stable:3.8.6`
- License: **GNU General Public License (GPL)**

References:
- Weka project site states Weka is issued under the GNU GPL: https://ml.cms.waikato.ac.nz/weka/  (see “issued under the GNU General Public License”)
- Maven Repository lists `weka-stable` license as GPL-3.0: https://mvnrepository.com/artifact/nz.ac.waikato.cms.weka/weka-stable

## Bounce (Weka third-party dependency)

- Component: `bounce`
- Version used by the Dockerfile: **0.18**
- Source: Maven Central coordinates `nz.ac.waikato.cms.weka.thirdparty:bounce:0.18`
- License: **BSD (as listed by Maven Central metadata)**

Reference:
- Sonatype Central metadata for bounce lists licenses as BSD: https://central.sonatype.com/artifact/nz.ac.waikato.cms.weka.thirdparty/bounce

## Notes

- These components are **not** redistributed as source in this repository; the Docker build downloads the JARs and places them in `/opt/weka/`.
- If you redistribute pre-built Docker images or packaged artifacts containing these JARs, ensure your distribution complies with the upstream licenses.
