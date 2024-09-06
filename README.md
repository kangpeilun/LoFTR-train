# This repo is creation for training LoFTR on custom dataset

This is initial LoFTR [README](docs/README.md) file.

As we all know, training LoFTR is not easy, I mean the official dont offer too much info for training our own custom datasets, so I give it a try.

This repo finishs the following things:

-[x] Using Colmap for sparse reconstruction.

-[x] Using Depth-Anything evaluating Depth for each image.

-[x] Convert depth.png to depth.h5 file.

-[x] Convert camera params to .npz file.

-[x] Automatically Generating train data struction.

-[x] Modify a little code because of the errors in the initial LoFTR.