# VarietySound
----

## VarietySound: Timbre-Controllable Video to Sound Generation via Unsupervised Information Disentanglement

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

### Timbre Controllable Video to Sound Generation

The current video-to-sound generation works share a common problem: all of their acoustic information comes from the model’s prediction and cannot control the timbre of the generated audio. To match this problem, we defined a task called Timbre Controllable Video to Sound Generation (TCVSG), whose target is to allow users to generate realistic sound effects with their desired timbre for silent videos.

![Existing Tasks & Proposed Task](demo/imgs/task.png)

we have a video clip V of an object breaking for movie production, but the natural recorded sounds are not impressive enough. So with this task, we can use an additional audio **A** with a more remarkable sound of breaking to generate an audio track for **V** . The generated audio **Aˆ** will be time-aligned with **V** , but has the same kind of sound as A which will make the video more exciting. As far as we know, we are the first to propose this task.

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

### Results Using Different Length Audio

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
