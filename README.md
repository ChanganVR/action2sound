# Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos

## Motivation
Generating realistic audio for human interactions is important for many applications, such as creating sound effects for films or virtual reality games. Existing approaches implicitly assume total correspondence between the video and audio during training, yet many sounds happen off-screen and have weak to no correspondence with the visuals---resulting in uncontrolled ambient sounds or hallucinations at test time. We propose a novel ambient-aware audio generation model, AV-LDM. We devise a novel audio-conditioning mechanism to learn to disentangle foreground action sounds from the ambient background sounds in in-the-wild training videos. Given a novel silent video, our model uses retrieval-augmented generation to create audio that matches the visual content both semantically and temporally. We train and evaluate our model on two in-the-wild egocentric video datasets Ego4D and EPIC-KITCHENS. Our model outperforms an array of existing methods, allows controllable generation of the ambient sound, and even shows promise for generalizing to computer graphics game clips. Overall, our work is the first to focus video-to-audio generation faithfully on the observed visual content despite training from uncurated clips with natural background sounds.

<p align="center">
  <img width="500" src="https://vision.cs.utexas.edu/projects/action2sound/static/images/concept.png" />
</p>

Project page: [Action2Sound](https://vision.cs.utexas.edu/projects/action2sound/)

## To-Do:
- Evaluation Code
- Training Code
- Pretrained weights
- Training code ☑️

## Installation 
1. Install this repo into pip by running the following command:
```
conda create -n action2sound
pip install -r requirements.txt
```

## Ego4D-Sounds Dataset
Prepare the dataset following this [link](https://github.com/Ego4DSounds/Ego4DSounds).

## Training and Testing
1. Training
```
python main.py --base configs/ldm/ego4dsounds.yaml -t --name audiocond --num-nodes 1 --gpus 0,1,2,3,4,5,6,7 --epoch 8 --scale_lr False --batch-size 90 --pretrained-model data/pretrained/av.pth --pool-patches max model.params.audio_cond_config.neighbor_audio_cond_prob=1 --audio-cond rand_neighbor
```
2. Testing
Add the following flags to the training command when training is done:
```
--test --eval-ckpt X --visualize
```

## Citation
If you find the code, data, or models useful for your research, please consider citing the following paper:
```
@article{chen2024action2sound,
    title = {Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos},
    author = {Changan Chen and Puyuan Peng and Ami Baid and Sherry Xue and Wei-Ning Hsu and David Harwath and Kristen Grauman},
    year = {2024},
    journal = {arXiv},
    }
```

## License
This repo is  licensed, as found in the [LICENSE](LICENSE) file.