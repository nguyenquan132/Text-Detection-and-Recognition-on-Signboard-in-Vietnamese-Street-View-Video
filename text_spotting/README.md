### Code Attribution

The following folders contain components adapted from official repositories:

* **DBNet** – adapted from the official repositories [PaddleOCR2Pytorch](../src/text_det/PaddleOCR2Pytorch/README.md) and [DBNet.pytorch](../src/text_det/DBNet.pytorch/README.MD)
* **TextPMs** – adapted from the official [TextPMs](../src/text_det/TextPMs/README.md) repository
* **OpenOCR** – adapted from the [OpenOCR](../src/text_rec/OpenOCR/README.md) repository

Selected modules were reused and reorganized for integration into this project.

### Evaluation
To evaluate text detection performance, use the following command:
```bash
python3 -m text_spotting.evaluation -g "path/to/ground_truth.zip" -s "path/to/results.zip"
```
For end-to-end text recognition evaluation with word spotting: 
```bash
python3 -m text_spotting.evaluation \
    -g "path/to/ground_truth.zip" \
    -s "path/to/results.zip" \
    -word_spotting \
    -case "insensitive" \
    -conf \
    -trans
```
**Arguments:**
- `-g`: Path to the **ground truth ZIP file** containing annotation files (.txt)
- `-s`: Path to the **results ZIP file** containing prediction files (.txt)