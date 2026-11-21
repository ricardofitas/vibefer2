# Reconstruct the original 0.1 and 0.2 builds

Start from a clean checkout of Johannes David's Vibefer at commit 94019ef77eb8a9e0680889ff63bda59ee5e42e8e. Apply vibefer2.patch to obtain the Vibefer2 0.1 sources. Apply vibefer2-0.2.patch on top to get the 0.2 baseline distributed at the root of this repository. Each patch is preserved byte-for-byte. The source object identifiers and SDK requirements are in [UPSTREAM.md](../UPSTREAM.md).

From the root of this Vibefer2 checkout, the following commands create a new adjacent checkout for the historical baseline:

    git clone https://github.com/JohannesDav/Vibefer.git ../vibefer-history
    git -C ../vibefer-history checkout --detach 94019ef77eb8a9e0680889ff63bda59ee5e42e8e
    git -C ../vibefer-history apply "$PWD/history/vibefer2.patch"

Build and measure 0.1 there first. Applying the following patch in that checkout moves the source to 0.2:

    git -C ../vibefer-history apply "$PWD/history/vibefer2-0.2.patch"
