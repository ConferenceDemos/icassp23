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

# VarietySound: Timbre-Controllable Video to Sound Generation via Unsupervised Information Disentanglement

----

## Abstract

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

## Timbre Controllable Video to Sound Generation

The current video-to-sound generation works share a common problem: all of their acoustic information comes from the model’s prediction and cannot control the timbre of the generated audio. To match this problem, we defined a task called Timbre Controllable Video to Sound Generation (TCVSG), whose target is to allow users to generate realistic sound effects with their desired timbre for silent videos.

![Existing Tasks & Proposed Task](demo/imgs/task.png)

we have a video clip V of an object breaking for movie production, but the natural recorded sounds are not impressive enough. So with this task, we can use an additional audio **A** with a more remarkable sound of breaking to generate an audio track for **V** . The generated audio **Aˆ** will be time-aligned with **V** , but has the same kind of sound as A which will make the video more exciting. As far as we know, we are the first to propose this task.

----

## Generated Results
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

## Method Detail

Our method is a process of information disentanglement and re-fusion.
We first disentangle the final audio information into three components: temporal information, timbre information, and background information, modeling them with three different encoders respectively, and then use a mel decoder to recombine these disentangled information for the reconstruction of the audio.
The disruption operations and the bottlenecks force the encoders to pass only the information that other encoders cannot supply, hence achieving the disentanglement; and with the sufficient information inputs, the mel decoder could finish the reconstruction of the target mel-spectrogram, hence achieving the re-fusion.
We also adopt the adversarial training, which helps the model to fit the distribution of the target mel-spectrogram better and thus obtain higher quality and better temporal alignment generation results.

### Information Components Describe

#### Temporal Information
The temporal information refers to the location information in the time sequence corresponding to the occurrence of the sound event.
In the temporal sequence, the position where the sound event occurs strongly correlates with the visual information adjacent to that position for real recorded video.
Therefore, in our method, this part of the information will be predicted using the visual feature sequence of the input video.
We also set a suitable bottleneck to ensure that the video can provide only temporal information without providing other acoustic content information.

#### Timbre Information
Timbre information is considered an acoustic characteristic inherent to the sound-producing object.
The distribution of timbres between different categories of objects can vary widely, and the timbres of different individuals of the same category of objects usually possess specific differences.
In our method, this part of the information will be predicted by the reference audio.
The random resampling transform refers to the operations of segmenting, expand-shrink transforming and random swapping of the tensor in the time sequence.
When encoding the reference timbre information, we perform a random resampling transform on the input reference audio in the time sequence to disrupt its temporal information.

#### Background Information
Background information is perceived as timbre-independent other acoustic information, such as background noise or off-screen background sound.
This part of the information is necessary for training to avoid model confusion due to the information mismatch.
We found that the energy of this part of the information is usually much smaller in the mel-spectrogram than the part where the timbre information is present.
Therefore, in the proposed method, we adopt an Energy Masking operation that masks the mel-spectrogram of the part of the energy larger than the median energy of the whole mel-spectrogram along the time dimension.
The energy masking operation discards both temporal and timbre-related information of the mel-spectrogram, preserving only the background information in the audio.
In the training phase, this information is added to match the target mel-spectrogram; in the inference phase, this information will be set to empty to generate clearer audio.


### Training and Inference

In the training phase of the generator, we use video features and mel-spectrogram from the same sample feed into the network, where the video features are fed into the Temporal Encoder and the mel-spectrogram is fed into the Acoustic and Background Encoders.
Our disentanglement method is unsupervised because there is no explicit intermediate information representation as a training target.
The outputs of the three encoders are jointly fed to the Mel Decoder to obtain the final reconstructed mel-spectrogram, and the generation losses are calculated with the real mel-spectrogram as Sec. \textit{Generator Loss} to guide the training of the generator.

In the training phase of the discriminator, for the Time-Domain Alignment Discriminator, the video features and mel-spectrogram from the same real sample are used as inputs to construct positive samples, while the real video features and the reconstructed mel-spectrogram are used as inputs to construct negative samples.
For the Multi-Window Mel Discriminator, the real mel-spectrogram from the sample is used as a positive sample and the reconstructed mel-spectrogram is used as a negative sample input.
The two discriminators calculate the losses and iteratively train according to the method in Sec. \textit{Discriminator Loss}.


In the inference phase, we feed the video features into the temporal encoder, the mel-spectrogram of the reference audio containing the target timbre into the acoustic encoder, and the mel-spectrogram of the muted audio into the background encoder, and then generate the sound through the mel decoder.
The choice of reference audio is arbitrary, depending on the desired target timbre.
Theoretically, the length of the video features and reference audio input during the inference phase is arbitrary, but it is necessary to ensure that the relevant events are present in the video and that the reference audio contains the desired timbre to obtain the normally generated sound.

----

## Model Structure and Configuration

### Self-Gated Acoustic Unit

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

### Model Configuration
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
    <td class="tg-nrix" rowspan="7">Temporal Domain <br>Aligment Discriminator</td>
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

## Detailed Experimental Result

### Baseline Model
Specifically, we build our cascade model using a video-to-sound generation model and a sound conversion model. The video-to-sound generation model is responsible for generating the corresponding audio for the muted video, while the sound conversion model is responsible for converting the timbre of the generated audio to the target timbre.
We chose [REGNET](https://github.com/PeihaoChen/regnet/) as the video-to-sound generation model, which has an excellent performance in the previous tasks.
For the sound conversion model, we consider using the voice conversion model which is used to accomplish similar tasks, since there is no explicitly defined sound conversion task and model.
We conducted some tests and found that some voice conversion models can perform simple sound conversion tasks.
Eventually, we chose unsupervised [SPEECHSPLIT](https://github.com/auspicious3000/SpeechSplit) as the sound conversion model because of the lack of detailed annotation of the timbres in each category in the dataset.
The cascade model is trained on the same dataset (VAS) and in the same environment, and inference is performed using the same test data.
In particular, instead of using speaker labels, our sound conversion model uses a sound embedding obtained from a learnable LSTM network as an alternative for providing target timbre information.
The cascade model's configuration follows the official implementation of the two models.

### Evaluation Design

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
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

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
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

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
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

### Sample Selection

To perform the evaluation, we randomly obtain test samples for each category in the following way.
Since our model accepts a video and a reference audio as a sample for input, we refer to it as a video-audio pair, and if the video and the audio are from the same raw video data, it will be called the original pair.

For audio realism evaluation, we will randomly select 10 data samples in the test set.
After breaking their original video-audio pairing relationship by random swapping, the new pair will be fed into the model to generate 10 samples, and then mixed these generated samples with the 10 ground truth samples to form the test samples.

For the temporal alignment evaluation, we will use the same method as above to obtain the generated samples and use the _ffmpeg_ tool to merge the audio samples with the corresponding video samples to produce the video samples with audio tracks.
We also mix 10 generated samples and 10 ground truth samples to form the test video samples in the temporal alignment evaluation.

For the timbre similarity evaluation, we randomly select 5 data samples in the test set, and for each audio sample, we combine them two-by-two with 3 random video samples to form 3 video-audio pairs, 15 in total.
The model takes these video-audio pairs as input and gets 3 generated samples for each reference audio, which will be used as test samples to compare with the reference audio.

For the ablation experiments, we only consider the reconstruction quality of the samples, so we randomly select 10 original pairs in the test set as input and obtain the generated samples.

### MOS Results
<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" colspan="10">MOS Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="2"><span style="font-weight:400;font-style:normal">Category</span></td>
    <td class="tg-wa1i" colspan="3"><span style="font-weight:400;font-style:normal">Audio Realism</span></td>
    <td class="tg-nrix" colspan="3"><span style="font-weight:400;font-style:normal">Temporal Alignment</span></td>
    <td class="tg-nrix" colspan="3"><span style="font-weight:400;font-style:normal">Timbre Similarity</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Ground Truth</td>
    <td class="tg-nrix">Baseline</td>
    <td class="tg-nrix">Proposed</td>
    <td class="tg-nrix">Ground Truth</td>
    <td class="tg-nrix">Baseline</td>
    <td class="tg-nrix">Proposed</td>
    <td class="tg-nrix">Ground Truth</td>
    <td class="tg-nrix">Baseline</td>
    <td class="tg-nrix">Proposed</td>
  </tr>
  <tr>
    <td class="tg-nrix">Baby</td>
    <td class="tg-nrix">$4.55 (\pm 0.10)$</td>
    <td class="tg-nrix">$2.67 (\pm 0.23)$</td>
    <td class="tg-nrix">$3.77 (\pm 0.15)$</td>
    <td class="tg-nrix">$4.43 (\pm 0.08)$</td>
    <td class="tg-nrix">$3.83 (\pm 0.17)$</td>
    <td class="tg-nrix">$4.07 (\pm 0.10)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">$3.46 (\pm 0.17)$</td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">$3.94 (\pm 0.08)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Cough</td>
    <td class="tg-nrix">$4.32 (\pm 0.11)$</td>
    <td class="tg-nrix">$3.30 (\pm 0.20)$</td>
    <td class="tg-nrix">$4.13(\pm 0.12)$</span></td>
    <td class="tg-nrix">$4.30(\pm 0.11)$</span></td>
    <td class="tg-nrix">$3.71(\pm 0.24)$</span></td>
    <td class="tg-nrix">$4.17(\pm 0.12)$</span></td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">$3.48 (\pm 0.22)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.59 (\pm 0.09)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Dog</td>
    <td class="tg-nrix">$4.45 (\pm 0.11)$</td>
    <td class="tg-nrix">$3.21 (\pm 0.19)$</td>
    <td class="tg-nrix">$4.18 (\pm 0.11)$</td>
    <td class="tg-nrix">$4.45 (\pm 0.08)$</td>
    <td class="tg-nrix">$4.32 (\pm 0.15)$</td>
    <td class="tg-nrix">$4.40 (\pm 0.08)$</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">$3.63 (\pm 0.15)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.09 (\pm 0.08)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Drum</td>
    <td class="tg-nrix">$4.62 (\pm 0.08)$</td>
    <td class="tg-nrix">$2.91 (\pm 0.21)$</td>
    <td class="tg-nrix">$4.12 (\pm 0.15)$</td>
    <td class="tg-nrix">$4.56 (\pm 0.06)$</td>
    <td class="tg-nrix">$3.64 (\pm 0.16)$</td>
    <td class="tg-nrix">$4.25 (\pm 0.11)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-nrix">$3.72 (\pm 0.13)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.85 (\pm 0.09)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Fireworks</td>
    <td class="tg-nrix">$4.56 (\pm 0.09)$</td>
    <td class="tg-nrix">$3.16 (\pm 0.22)$</td>
    <td class="tg-nrix">$4.23 (\pm 0.13)$</td>
    <td class="tg-nrix">$4.47 (\pm 0.08)$</td>
    <td class="tg-nrix">$4.00 (\pm 0.21)$</td>
    <td class="tg-nrix">$4.35 (\pm 0.10)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-nrix">$3.43 (\pm 0.25)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.93 (\pm 0.07)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Gun</td>
    <td class="tg-nrix">$4.38 (\pm 0.12)$</td>
    <td class="tg-nrix">$2.76 (\pm 0.22)$</td>
    <td class="tg-nrix">$4.02 (\pm 0.15)$</td>
    <td class="tg-nrix">$4.45 (\pm 0.09)$</td>
    <td class="tg-nrix">$4.08 (\pm 0.17)$</td>
    <td class="tg-nrix">$4.25 (\pm 0.12)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-nrix">$3.45 (\pm 0.18)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.98 (\pm 0.08)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Hammer</td>
    <td class="tg-nrix">$4.43 (\pm 0.12)$</td>
    <td class="tg-nrix">$3.16 (\pm 0.26)$</td>
    <td class="tg-nrix">$3.84 (\pm 0.14)$</td>
    <td class="tg-nrix">$4.31 (\pm 0.08)$</td>
    <td class="tg-nrix">$3.88 (\pm 0.19)$</td>
    <td class="tg-nrix">$4.19 (\pm 0.13)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-nrix">$3.74 (\pm 0.17)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.99 (\pm 0.10)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Sneeze</td>
    <td class="tg-nrix">$4.04 (\pm 0.13)$</td>
    <td class="tg-nrix">$2.75 (\pm 0.22)$</td>
    <td class="tg-nrix">$4.00 (\pm 0.15)$</td>
    <td class="tg-nrix">$4.28 (\pm 0.12)$</td>
    <td class="tg-nrix">$3.76 (\pm 0.23)$</td>
    <td class="tg-nrix">$4.16 (\pm 0.11)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-nrix">$3.62 (\pm 0.18)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.72 (\pm 0.08)$</span></td>
  </tr>
  <tr>
    <td class="tg-wa1i">Average</td>
    <td class="tg-nrix">$4.42(\pm 0.04)$</td>
    <td class="tg-nrix">$2.99 (\pm 0.08)$</td>
    <td class="tg-nrix">$4.04(\pm 0.05)$</td>
    <td class="tg-nrix">$4.41(\pm 0.03)$</td>
    <td class="tg-nrix">$3.90 (\pm 0.07)$</td>
    <td class="tg-nrix">$4.23(\pm 0.04)$</td>
    <td class="tg-baqh">-</td>
    <td class="tg-nrix">$3.57 (\pm 0.07)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.89 (\pm 0.03)$</span></td>
  </tr>
</tbody>
</table>

### Cosine Similarity

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2"><span style="font-weight:400;font-style:normal">Category</span></th>
    <th class="tg-nrix" colspan="2">Timbre Cosine Similarity</th>
  </tr>
  <tr>
    <th class="tg-nrix">Baseline</th>
    <th class="tg-nrix">Proposed</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix">Baby</td>
    <td class="tg-nrix">$0.86 (\pm 0.01)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.88 (\pm 0.00)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Cough</td>
    <td class="tg-nrix">$0.86 (\pm 0.00)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.93 (\pm 0.01)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Dog</td>
    <td class="tg-nrix">$0.77 (\pm 0.00)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.96 (\pm 0.00)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Drum</td>
    <td class="tg-nrix">$0.74 (\pm 0.03)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.84 (\pm 0.00)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Fireworks</td>
    <td class="tg-nrix">$0.88 (\pm 0.01)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.89 (\pm 0.01)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Gun</td>
    <td class="tg-nrix">$0.83 (\pm 0.01)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.88 (\pm 0.01)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Hammer</td>
    <td class="tg-nrix">$0.80 (\pm 0.02)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.89 (\pm 0.02)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Sneeze</td>
    <td class="tg-nrix">$0.88 (\pm 0.01)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">0.98 $(\pm 0.01)$</span></td>
  </tr>
  <tr>
    <td class="tg-nrix">Average</td>
    <td class="tg-nrix">$0.84 (\pm 0.01)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$0.90 (\pm 0.01)$</span></td>
  </tr>
</tbody>
</table>

Through the third-party evaluation on the _Amazon Mechanical Turk_ (AMT), we obtained the evaluation results of our model.

As shown in the Table \ref{tab:Evaluation}, the proposed model achieves scores closer to ground truth in terms of both audio realism and temporal alignment by comparing with the baseline model.
The category of _Dog_ and _Fireworks_ have the best average performance in the two evaluations.
The category of _Baby_ gains the worst performance in the evaluation of audio realism and temporal alignment due to the uncertainty and diversity in human behavior which is hard for modeling, the same trend also appears in the category of _Cough_ and _Sneeze_.
Due to the imbalance in the amount of data in each category in the dataset, we can see that the four categories with smaller amounts of data (_Cough, Gun, Hammer_ and _Sneeze_) will have overall lower temporal alignment scores than the four categories with larger amounts of data (_Baby, Dog, Fireworks_ and _Drum_) in both evaluations, suggesting that the modeling of temporal alignment may be more sensitive to the amount of data.

In the evaluation of the audio quality, the baseline model achieved a relatively low score. 
This is because the cascade model accumulates the errors of both models during the generation process, bringing apparent defects to the generated audio, such as noise, electrotonality, or mechanicalness.

For the similarity of the timbre, as shown in Table \ref{tab:sim}, the proposed model achieve higher scores both in the subjective and objective evaluation, which means the result of proposed model have a timbre closer to the ground truth than the baseline model.

We did not compare the generation speed because, empirically, the inference efficiency of a single model is usually much higher than that of a cascade model.

As a summary, by obtaining generation results and subjective evaluation results above, we have successfully demonstrated the effectiveness of our method and model.



### Ablation Results
<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-xxp7{font-style:italic;font-weight:bold;text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" colspan="7"><span style="font-weight:400;font-style:normal">MCD Scores</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="2"><span style="font-weight:400;font-style:normal">Category</span></td>
    <td class="tg-nrix" rowspan="2">Proposed</td>
    <td class="tg-nrix" colspan="3">Generator Encoder Ablation</td>
    <td class="tg-nrix" colspan="2">Discriminator Ablation</td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">w/o Temporal</span></td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">w/o Timbre</span></td>
    <td class="tg-nrix">w/o Background</td>
    <td class="tg-nrix">w/o Multi-Window <br>Mel Discriminator</td>
    <td class="tg-nrix">w/o Temporal Domain <br>Alignment Discriminator</td>
  </tr>
  <tr>
    <td class="tg-baqh">Baby</td>
    <td class="tg-wa1i">$3.75(\pm 0.25)$</td>
    <td class="tg-nrix">$5.26(\pm 0.16)$</td>
    <td class="tg-nrix">$6.84(\pm 0.26)$</td>
    <td class="tg-nrix">$4.77(\pm 0.10)$</td>
    <td class="tg-nrix">$5.29(\pm 0.12)$</td>
    <td class="tg-nrix">$5.12(\pm 0.12)$</td>
  </tr>
  <tr>
    <td class="tg-baqh">Cough</td>
    <td class="tg-wa1i">$3.72(\pm 0.14)$</td>
    <td class="tg-nrix">$3.87(\pm 0.23)$</td>
    <td class="tg-nrix">$5.16(\pm 0.20)$</td>
    <td class="tg-nrix">$3.77(\pm 0.23)$</td>
    <td class="tg-nrix">$4.41(\pm 0.17)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.79(\pm 0.19)$</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">Dog</td>
    <td class="tg-wa1i">$3.90(\pm 0.14)$</td>
    <td class="tg-nrix">$4.42(\pm 0.13)$</td>
    <td class="tg-nrix">$4.89(\pm 0.15)$</td>
    <td class="tg-nrix">$4.19(\pm 0.18)$</td>
    <td class="tg-nrix">$4.22(\pm 0.20)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.44(\pm 0.14)$</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">Drum</td>
    <td class="tg-wa1i">$3.38(\pm 0.16)$</td>
    <td class="tg-nrix">$4.10(\pm 0.18)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$5.36(\pm 0.30)$</span></td>
    <td class="tg-nrix">$4.01(\pm 0.18)$</td>
    <td class="tg-nrix">$4.11(\pm 0.21)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.38(\pm 0.26)$</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">Fireworks</td>
    <td class="tg-wa1i">$3.10(\pm 0.13)$</td>
    <td class="tg-nrix">$3.85(\pm 0.08)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.91(\pm 0.18)$</span></td>
    <td class="tg-nrix">$3.64(\pm 0.10)$</td>
    <td class="tg-nrix">$3.61(\pm 0.10)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.68(\pm 0.12)$</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">Gun</td>
    <td class="tg-wa1i">$3.39(\pm 0.12)$</td>
    <td class="tg-nrix">$3.88(\pm 0.12)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.74(\pm 0.21)$</span></td>
    <td class="tg-nrix">$3.73(\pm 0.20)$</td>
    <td class="tg-nrix">$3.73(\pm 0.14)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$3.77(\pm 0.15)$</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">Hammer</td>
    <td class="tg-wa1i">$3.47(\pm 0.16)$</td>
    <td class="tg-nrix">$4.95(\pm 0.14)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.53(\pm 0.17)$</span></td>
    <td class="tg-nrix">$4.05(\pm 0.27)$</td>
    <td class="tg-nrix">$4.02(\pm 0.17)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.39(\pm 0.13)$</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">Sneeze</td>
    <td class="tg-wa1i">$4.04(\pm 0.18)$</td>
    <td class="tg-nrix">$4.50(\pm 0.15)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$5.91(\pm 0.38)$</span></td>
    <td class="tg-nrix">$4.34(\pm 0.13)$</td>
    <td class="tg-nrix">$4.58(\pm 0.22)$</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">$4.97(\pm 0.20)$</span></td>
  </tr>
  <tr>
    <td class="tg-amwm">Average</td>
    <td class="tg-xxp7">$3.60(\pm 0.07)$</td>
    <td class="tg-wa1i">$4.33(\pm 0.08)$</td>
    <td class="tg-wa1i"><span style="font-style:normal">$5.25(\pm 0.12)$</span></td>
    <td class="tg-wa1i">$4.14(\pm 0.07)$</td>
    <td class="tg-wa1i">$4.25(\pm 0.08)$</td>
    <td class="tg-wa1i"><span style="font-style:normal">$4.43(\pm 0.08)$</span></td>
  </tr>
</tbody>
</table>
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
