# SCH
Source code for TPAMI paper "Cross-Modal Hashing Method with Properties of Hamming Space: A New Perspective"

## Datasets
Please refer to the provided [link](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_tensorflow/DCMH_tensorflow) to download the dataset, create a data folder and update data path in settings.py.

## Train model

You can directly run the file 
```
python train.py --Bit 16 --GID 0 --DS 0
```
to get the results.

## Evaluate the model

Modify the settings.py line 7
```
EVAL = True
```

You can downlod the trained models via following links (have been updated):

| Dataset | Hash Bit | Downlod |
| :-- | :--: | :--: |
| MIR | 16 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EZKBGl3D-sxHiEWSDPFt_aMB8qNdueWAVlSqif0eIlD2SQ?e=GBUwiT) |
| MIR | 32 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EQNW4xO59cJArBApgqXP80gBHd-IfAikre5O8-cqbvC_kw?e=86Eq2D) |
| MIR | 64 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EQDAoKNVoXRDscnPEhvRZvIBmgKyrsdV0QnRrqN8KzVD_Q?e=EvpsYV) |
| MIR | 128 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EbKID3jFq1RGiiiAwJYRCIEBtItd0t9dA2r8nVv7iM8pZQ?e=ZNiE60) |
| NUS | 16 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EU-nBOz5rCpKs9Df9JHhhbABr1u-7RXhaxRH6OtABPOiSA?e=nCFLF1) |
| NUS | 32 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EWp3d2dGaihHquexIes01CUBrVoVGPyAfnR9PEO8RXpOKw?e=bMRpD7) |
| NUS | 64 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/EQEF3zFiFLVJi85KpwrzE48BSNn1wj6spRuvJtS1ujLpwA?e=0BbbMW) |
| NUS | 128 | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/20481446_life_hkbu_edu_hk/Edx0u9xWrrFBnRzqo_aHOUgBlrmxAQ8tiMQTOfA1uJQgqg?e=dkL0gD) |

## Citation
If you find SCH useful in your research, please consider citing:

```
@article{hu2024cross,
  title={Cross-Modal Hashing Method with Properties of Hamming Space: A New Perspective},
  author={Hu, Zhikai and Cheung, Yiu-ming and Li, Mengke and Lan, Weichao and Zhang, Donglin and Liu, Qiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

