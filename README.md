<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# VarietySound
----

## VarietySound: Timbre-Controllable Video to Sound Generation via Unsupervised Information Disentanglement

----

### Abstract

Video to sound generation aims to generate realistic and natural sound given a video input.
However, previous video-to-sound generation methods can only generate a random or average timbre without any controls or specializations of the generated sound timbre, leading to the problem that people cannot obtain the desired timbre under these methods sometimes. 
In this paper, we pose the task of generating sound with a specific timbre given a video input and a reference audio sample.
To solve this task, we disentangle each target sound audio into three components: temporal information, acoustic information, and background information.
We first use three encoders to encode these components respectively:
- 1) a temporal encoder to encode temporal information, which is fed with video frames since the input video shares the same temporal information as the original audio;
- 2) an acoustic encoder to encode timbre information, which takes the original audio as input and discards its temporal information by a temporal-corrupting operation;
- 3) and a background encoder to encode the residual or background sound, which uses the background part of the original audio as input.

Then we use a decoder to reconstruct the audio given these disentangled representations encoded by three encoders.
To make the generated result achieve better quality and temporal alignment, we also adopt a mel discriminator and a temporal discriminator for the adversarial training.
In inference, we feed the video, the reference audio and the silent audio into temporal, acoustic and background encoders and then generate the audio which is synchronized with the events in the video and has the same acoustic characteristics as the reference audio with no background noise.
Our experimental results on the VAS dataset demonstrate that our method can generate high-quality audio samples with good synchronization with events in video and high timbre similarity with the reference audio.

----

### Timbre Controllable Video to Sound Generation

The current video-to-sound generation works share a common problem: all of their acoustic information comes from the model’s prediction and cannot control the timbre of the generated audio. To match this problem, we defined a task called Timbre Controllable Video to Sound Generation (TCVSG), whose target is to allow users to generate realistic sound effects with their desired timbre for silent videos.

![Existing Tasks & Proposed Task](demo/imgs/task.png)

we have a video clip V of an object breaking for movie production, but the natural recorded sounds are not impressive enough. So with this task, we can use an additional audio **A** with a more remarkable sound of breaking to generate an audio track for **V** . The generated audio **Aˆ** will be time-aligned with **V** , but has the same kind of sound as A which will make the video more exciting. As far as we know, we are the first to propose this task.

----

### Generated Results
<table>
    <thead>
        <th>Category</th>
        <th>Original Video</th>
        <th>Reference Audio</th>
        <th>Baseline</th>
        <th>Proposed</th>
    </thead>
    <tbody>
        <tr>
            <th>Baby</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/baby.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/baby.wav" type="audio/wav"></audio></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/baby_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/baby_gen.mp4" type="video/mp4"></video></td>
        </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Cough</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/cough.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/cough.wav" type="audio/wav"></audio></td>
          <td><video width="100%" height="240" controls><source src="demo/videos/cough_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/cough_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Dog</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/dog.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/dog.wav" type="audio/wav"></audio></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/dog_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/dog_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Drum</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/drum.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/drum.wav" type="audio/wav"></audio></td>
          <td><video width="100%" height="240" controls><source src="demo/videos/drum_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/drum_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Fireworks</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/fireworks.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/fireworks.wav" type="audio/wav"></audio></td>
          <td><video width="100%" height="240" controls><source src="demo/videos/fireworks_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/fireworks_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Gun</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/gun.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/gun.wav" type="audio/wav"></audio></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/gun_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/gun_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Hammer</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/hammer.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/hammer.wav" type="audio/wav"></audio></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/hammer_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/hammer_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
    <tbody>
      <tr>
          <th>Sneeze</th>
            <td><video width="100%" height="240" controls><source src="demo/videos/sneeze.mp4" type="video/mp4"></video></td>  
            <td><audio controls style="width: 150px;"><source src="demo/audios/sneeze.wav" type="audio/wav"></audio></td>
          <td><video width="100%" height="240" controls><source src="demo/videos/sneeze_baseline.mp4" type="video/mp4"></video></td>
            <td><video width="100%" height="240" controls><source src="demo/videos/sneeze_gen.mp4" type="video/mp4"></video></td>
      </tr>
    </tbody>
</table>


----

### Detailed Model Structure and Configuration

#### Self-Gated Acoustic Unit

Each SGAU has two inputs and two outputs, which we call feature inputs, feature outputs, conditional inputs, and conditional outputs, respectively.
The feature input receives the input vectors and passes through two layers of 1D convolutional layers, which we call the input gate, and then normalized by Instance Normalization.
The conditional inputs receive the input vectors and then pass through two single 1D convolutional layers, which we call the output gate and skip gate.
The output gate and the skip gate are both normalized by Group Normalization.
The output of the jump gate is used as the output vector of the conditional output after the random resampling transform.
Meanwhile, the output of the input gate is added with the output of the skip gate and the output gate, respectively, and then multiplied after different activation functions.
The above result is transformed by Random Resampling and used as the output vector of the feature output.

For a clearer statement, the gated unit is described by the following equations:

$\mathbf{x_{o}}=\boldsymbol{R}[\tanh(\boldsymbol{W_{s}} * \mathbf{c_{i}}+\boldsymbol{V_{i}} * \mathbf{x_{i}}) \odot \sigma(\boldsymbol{W_{o}} *  \mathbf{c_{i}}+\boldsymbol{V_{i}} * \mathbf{x_{i}})]$

$\mathbf{c_{o}}=\boldsymbol{R}[\boldsymbol{W_{s}} * \mathbf{c_{i}}]$

where $\mathbf{x_{i}}$ and $\mathbf{c_{i}}$ denote two inputs of the unit, $\mathbf{x_{o}}$ and $\mathbf{c_{o}}$ denote two outputs of the unit. $\odot$ denotes an element-wise multiplication operator, $\sigma(\cdot)$ is a sigmoid function. $\boldsymbol{R}[ \cdot ]$ denotes the random resampling transform, 
$\boldsymbol{W\_{\cdot}}\* $ and $\boldsymbol{V\_{\cdot}}\* $ denote the single layer convolution in skip or output gate and the 2-layer convolutions in input gate separately.

#### Model Configuration
We list hyperparameters and configurations of all models used in our experiments in Table:
<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-wa1i">Module</th>
    <th class="tg-uzvj">Hyperparameter<br></th>
    <th class="tg-uzvj">Size</th>
    <th class="tg-wa1i">Module</th>
    <th class="tg-wa1i">Hyperparameter<br></th>
    <th class="tg-wa1i">Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="7"><span style="font-weight:400;font-style:normal">Temporal </span><br><span style="font-weight:400;font-style:normal">Encoder</span></td>
    <td class="tg-9wq8"> Input Dimension </td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal">2048</span></td>
    <td class="tg-nrix" rowspan="7">Self-Gated <br>Acoustic Unit</td>
    <td class="tg-nrix">Input-Gate <br>Conv1D-1 Kernel</td>
    <td class="tg-nrix">5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Conv1D Layers</td>
    <td class="tg-9wq8">8</td>
    <td class="tg-nrix">Input-Gate <br>Conv1D-2 Kernel</td>
    <td class="tg-nrix">7</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Conv1D Kernel</td>
    <td class="tg-9wq8">5</td>
    <td class="tg-nrix">Input-Gate<br>Filter Size</td>
    <td class="tg-nrix">512</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Conv1D Filter Size</td>
    <td class="tg-9wq8">512</td>
    <td class="tg-nrix">Output-Gate <br>Conv1D Kernel</td>
    <td class="tg-nrix">3</td>
  </tr>
  <tr>
    <td class="tg-9wq8">LSTM Layers</td>
    <td class="tg-9wq8">2</td>
    <td class="tg-nrix">Output-Gate <br>Conv1D Filter Size</td>
    <td class="tg-nrix">512</td>
  </tr>
  <tr>
    <td class="tg-9wq8">LSTM Hidden Size</td>
    <td class="tg-9wq8">256</td>
    <td class="tg-nrix">Skip-Gate <br>Conv1D Kernel</td>
    <td class="tg-nrix">5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Output Dimension</td>
    <td class="tg-9wq8">8</td>
    <td class="tg-nrix">Skip-Gate <br>Conv1D Filter Size</td>
    <td class="tg-nrix">512</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">Acoustic <br>Encoder</td>
    <td class="tg-9wq8">SGAU Layers</td>
    <td class="tg-9wq8">5</td>
    <td class="tg-nrix" rowspan="5">Mel Decoder</td>
    <td class="tg-nrix">ConvT1D Layers</td>
    <td class="tg-nrix">2</td>
  </tr>
  <tr>
    <td class="tg-9wq8">LSTM Layers</td>
    <td class="tg-9wq8">2</td>
    <td class="tg-nrix">ConvT1D Kernel</td>
    <td class="tg-nrix">4</td>
  </tr>
  <tr>
    <td class="tg-9wq8">LSTM Hidden</td>
    <td class="tg-9wq8">256</td>
    <td class="tg-nrix">ConvT1D Stride</td>
    <td class="tg-nrix">2</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="7">Time-Domain <br>Aligment Discriminator</td>
    <td class="tg-9wq8">ConvT1D Layers</td>
    <td class="tg-9wq8">2</td>
    <td class="tg-nrix">ConvT1D <br>Filter Size</td>
    <td class="tg-nrix">1024</td>
  </tr>
  <tr>
    <td class="tg-9wq8">ConvT1D Kernel</td>
    <td class="tg-9wq8">4</td>
    <td class="tg-nrix">FFT Blocks</td>
    <td class="tg-nrix">4</td>
  </tr>
  <tr>
    <td class="tg-9wq8">ConvT1D Stride</td>
    <td class="tg-9wq8">2</td>
    <td class="tg-nrix" rowspan="2">Background <br>Encoder</td>
    <td class="tg-nrix">LSTM Layers</td>
    <td class="tg-nrix">2</td>
  </tr>
  <tr>
    <td class="tg-9wq8">ConvT1D Filter Size</td>
    <td class="tg-9wq8">1024</td>
    <td class="tg-baqh">LSTM Hidden</td>
    <td class="tg-baqh">128</td>
  </tr>
  <tr>
    <td class="tg-nrix">Conv1D Layers</td>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix" rowspan="4">FFT Block</td>
    <td class="tg-nrix">Hidden Size</td>
    <td class="tg-nrix">512</td>
  </tr>
  <tr>
    <td class="tg-nrix">Conv1D Kernel</td>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">Attention Headers</td>
    <td class="tg-nrix">2</td>
  </tr>
  <tr>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">Conv1D Filter Size</span></td>
    <td class="tg-nrix">512</td>
    <td class="tg-nrix">Conv1D Kernel</td>
    <td class="tg-nrix">9</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="2">Training Loss</td>
    <td class="tg-nrix">$\lambda_{m}$</td>
    <td class="tg-nrix">1e5</td>
    <td class="tg-nrix">Conv1D Filter Size</td>
    <td class="tg-nrix">512</td>
  </tr>
  <tr>
    <td class="tg-nrix">$\lambda_{a}$</td>
    <td class="tg-nrix">1.0</td>
    <td class="tg-nrix"></td>
    <td class="tg-nrix"></td>
    <td class="tg-nrix"></td>
  </tr>
</tbody>
</table>

----

### Detailed Experimental Result

#### Baseline Model
Specifically, we build our cascade model using a video-to-sound generation model and a sound conversion model. The video-to-sound generation model is responsible for generating the corresponding audio for the muted video, while the sound conversion model is responsible for converting the timbre of the generated audio to the target timbre.
We chose [REGNET](https://github.com/PeihaoChen/regnet/) as the video-to-sound generation model, which has an excellent performance in the previous tasks.
For the sound conversion model, we consider using the voice conversion model which is used to accomplish similar tasks, since there is no explicitly defined sound conversion task and model.
We conducted some tests and found that some voice conversion models can perform simple sound conversion tasks.
Eventually, we chose unsupervised [SPEECHSPLIT](https://github.com/auspicious3000/SpeechSplit) as the sound conversion model because of the lack of detailed annotation of the timbres in each category in the dataset.
The cascade model is trained on the same dataset (VAS) and in the same environment, and inference is performed using the same test data.
In particular, instead of using speaker labels, our sound conversion model uses a sound embedding obtained from a learnable LSTM network as an alternative for providing target timbre information.
The cascade model's configuration follows the official implementation of the two models.

#### Evaluation Design

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" colspan="2">MOS of Realism</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix">Score</td>
    <td class="tg-nrix">Meaning</td>
  </tr>
  <tr>
    <td class="tg-nrix">5</td>
    <td class="tg-nrix">Completely real sound</td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">Mostly real sound</td>
  </tr>
  <tr>
    <td class="tg-nrix">3</td>
    <td class="tg-nrix">Equally real and unreal sound</td>
  </tr>
  <tr>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">Mostly unreal sound</td>
  </tr>
  <tr>
    <td class="tg-nrix">1</td>
    <td class="tg-nrix">Completely unreal sound</td>
  </tr>
</tbody>
</table>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" colspan="2">MOS of Temporal Alignment</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix">Score</td>
    <td class="tg-nrix">Meaning</td>
  </tr>
  <tr>
    <td class="tg-nrix">5</td>
    <td class="tg-nrix">Events in video and events in audio<br> occur simultaneously</td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">Slight misalignment between events<br> in video and events in audio</td>
  </tr>
  <tr>
    <td class="tg-nrix">3</td>
    <td class="tg-nrix">Exist misalignment in some positions</td>
  </tr>
  <tr>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">Exist misalignment in most of the positions</td>
  </tr>
  <tr>
    <td class="tg-nrix">1</td>
    <td class="tg-nrix">Completely misalignment, no events in audio<br> can match the video</td>
  </tr>
</tbody>
</table>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" colspan="2">MOS of Timbre Similarity</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix">Score</td>
    <td class="tg-nrix">Meaning</td>
  </tr>
  <tr>
    <td class="tg-nrix">5</td>
    <td class="tg-nrix">Timbre is exactly the same as target</td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">Timbre has high similarity<br> with the target but not same</td>
  </tr>
  <tr>
    <td class="tg-nrix">3</td>
    <td class="tg-nrix">Timbre has similarity with the target,<br> but there are obvious differences</td>
  </tr>
  <tr>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">Timbre has a large gap with the target,<br> but share the same category of the sound</td>
  </tr>
  <tr>
    <td class="tg-nrix">1</td>
    <td class="tg-nrix">Timbre is completely different to the target</td>
  </tr>
</tbody>
</table>

We give the detailed definition of the MOS score on the subjective evaluation of audio realism, temporal alignment and timbre similarity in Table above, respectively.

In the evaluation of the realism of the generated audio, we will ask the raters to listen to several test audios and rate the realistic level of the audio content.
The higher the score, the more realistic the generated audio.

In the evaluation of temporal alignment, we ask the raters to watch several test videos with their audio and rate the alignment of them.
Samples with a shorter interval between the moment of the event in the video and the moment of the corresponding audio event will receive a higher score.

In the evaluation of timbre similarity, we ask the rater to listen to one original audio and several test audios and score how similar the test audio timbre is to the original audio timbre.
For cosine similarity, we calculate the cosine similarity between the target audio and ground truth audio timbre features using the following equation:

$CosSim(X,Y) = \frac{ \sum \limits_{i=1}^{n}(x_{i} * y_{i})}{ \sqrt{ \sum \limits_{i=1}^{n}(x_{i})^{2}} \sqrt{  \sum \limits_{i=1}^{n}(y_{i})^{2} } } $

, and the timbre features are calculated by the [third-party library](https://github.com/resemble-ai/Resemblyzer).
The higher the similarity between the test audio and the original audio sound, the higher the score will be.

#### Sample Selection

To perform the evaluation, we randomly obtain test samples for each category in the following way.
Since our model accepts a video and a reference audio as a sample for input, we refer to it as a video-audio pair, and if the video and the audio are from the same raw video data, it will be called the original pair.

For audio realism evaluation, we will randomly select 10 data samples in the test set.
After breaking their original video-audio pairing relationship by random swapping, the new pair will be fed into the model to generate 10 samples, and then mixed these generated samples with the 10 ground truth samples to form the test samples.

For the temporal alignment evaluation, we will use the same method as above to obtain the generated samples and use the _ffmpeg_ tool to merge the audio samples with the corresponding video samples to produce the video samples with audio tracks.
We also mix 10 generated samples and 10 ground truth samples to form the test video samples in the temporal alignment evaluation.

For the timbre similarity evaluation, we randomly select 5 data samples in the test set, and for each audio sample, we combine them two-by-two with 3 random video samples to form 3 video-audio pairs, 15 in total.
The model takes these video-audio pairs as input and gets 3 generated samples for each reference audio, which will be used as test samples to compare with the reference audio.

For the ablation experiments, we only consider the reconstruction quality of the samples, so we randomly select 10 original pairs in the test set as input and obtain the generated samples.

#### MOS Results

#### Cosine Similarity

#### Ablation Results

In the ablation experiment of the generator, we calculate the MCD scores for the generated result when one information component encoded by a certain encoder is removed as an objective evaluation.
As shown in Table above, the generated results of our model achieve the second-lowest score on all experiments.
There is a phenomeon may evoke a confuse,
The result is reasonable since there is a trade-off between audio reconstruction quality and audio quality due to the presence of background noise.
Specifically, the three parts of information are all necessary for the reconstruction of the target mel-spectrogram, and it will gain a larger distance between the generated result and the target audio when we discard one of the information, even if the information may conduct a negative impact on the quality of our generated results (e.g., the background noise).
Meanwhile, the above results can also corroborate the effectiveness of the background encoder in our model.
The results also show that timbre information has a more significant impact on the quality of the reconstructed audio than temporal information on average.

The generated result of our model acquires the minimum MCD score, which has successfully demonstrated the effectiveness of the encoders of our model.
To better illustrate the above experimental results, we compare the reconstructed mel-spectrogram when one information component is removed or added (by setting the input vector to Gaussian noise or zero), and visualize the mel-spectrogram reconstruction results as shown in Figure above.
As can be observed, when the temporal information is removed, the output mel-spectrogram becomes meaningless content that is uniformly distributed over the time series, and when the timbre information is removed, the output becomes a random timbre without specific spectral characteristics.
For the background information, when it is added during inference, the mel-spectrogram's background noise becomes brighter.

In the ablation experiments of discriminators, we retrain our model with one of the discriminators disabled. The experiments are performed under the same settings and configurations as before.  
As can be observed in Table , the MCD scores of the generated results for almost all categories decreased to various extents after removing any of the discriminators.
On average, the impact of removing the Temporal Domain Alignment Discriminator is more significant than that of removing the Multi-window Mel Discriminator.
Due to the fact that the mel-spectrogram compresses the high-frequency components to some extent, some of the categories with high-frequency information content, such as _Fireworks_, _Gun_, and _Hammer_, do not have significant differences in the scores obtained after removing the mel discriminator.

#### Results Using Different Length Audio

(Take Dogs For Example)

<table>
    <thead>
        <th>Sample Number</th>
        <th>Ground Truth Audio From Video Sound Track</th>
        <th>Reference Audio</th>
        <th>Using 0.5x Length Reference Audio</th>
        <th>Using 1.0x Length Reference Audio</th>
        <th>Using 2.0x Length Reference Audio</th>
    </thead>
    <tbody>
        <tr>
            <th>Sample 1</th>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/gt/1.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/ref.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/0.5/1.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/1.0/1.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/2.0/1.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>Sample 2</th>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/gt/2.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/ref.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/0.5/2.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/1.0/2.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/2.0/2.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>Sample 3</th>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/gt/3.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/ref.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/0.5/3.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/1.0/3.wav" type="audio/wav"></audio></td>
            <td><audio controls style="width: 150px;"><source src="demo/audios/lengthdemos/2.0/3.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
</table>

----

<!-- ## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/VarietySound/Anonymous1398/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/VarietySound/Anonymous1398/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
 -->
