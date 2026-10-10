# Plant Disease Detection

Classifies plant leaf images into 38 crop-disease/healthy classes using transfer learning with MobileNetV2.

## Dataset
[PlantVillage](https://huggingface.co/datasets/geraldmc/plantvillage-full) (54,306 leaf images, 38 classes), loaded from the Hugging Face Hub. Uses its leaf-grouped train/test split so images of the same physical leaf never appear in both sets; 10% of training leaves are held out for validation.

## Approach
1. MobileNetV2 pretrained on ImageNet as the feature extractor
2. Stage 1: train a new classifier head (frozen base)
3. Stage 2: fine-tune the top 30 layers at a low learning rate
4. Data augmentation: flips, rotation, zoom, contrast

## Results
- Test accuracy: **88.7%** (fill in after training)
- See `outputs/training_curves.png`, `outputs/confusion_matrix.png`, `outputs/classification_report.txt`

## Limitations
PlantVillage images are taken in controlled conditions (plain background). Accuracy on real field photos will be lower.

## Run it
```bash
pip install -r requirements.txt
python train.py                    # trains and saves the model to ./outputs
python predict.py leaf.jpg         # predict one image
streamlit run app.py               # web interface
```
