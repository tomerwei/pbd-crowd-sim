# pbd-crowd-sim

## Position-Based Real-Time Simulation of Large Crowds

[Position-Based Real-Time Simulation of Large Crowds](http://www.cs.ucla.edu/~tweiss)<br />
[Tomer Weiss](http://www.cs.ucla.edu/~tweiss), Alan Litteneker, Chenfanfu Jiang, and Demetri Terzopoulos<br/>
University of California, Los Angeles

# Instructions
This software has been tested on macOS Mojave. <br/>
<!-- a normal html comment 
It depends on [Eigen](eigen.tuxfamily.org/) and [callisto](www.nieuwenhuisen.nl/callisto/), 
which are included in the code. Callisto is used for visualization purposes and requires DirectX 9.0c. 
If you want to compile the code with x64 support, you should disable the visualizer. <br/>
--> 

After the code is compiled, to simulate a scenario go to the top folder and run:</br>
"./makecrowds" <br/>

# TODO
* Add more scenarios
* Modularize visualization from simulation code 
* Add GPU code 

# Citation
<p>BibTex:</p>
<pre><code>
@article{weiss2019position,
  title={Position-based real-time simulation of large crowds},
  author={Weiss, Tomer and Litteneker, Alan and Jiang, Chenfanfu and Terzopoulos, Demetri},
  journal={Computers \& Graphics},
  volume={78},
  pages={12--22},
  year={2019},
  publisher={Elsevier}
}
</code></pre>



