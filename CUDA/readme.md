## How to run CUDA on Google Colab
- **To install CUDA plugin**
<pre>
<b>!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git</b>
<b>%load_ext nvcc_plugin</b>
</pre>
- **Code format**
<pre>
<b>%%cu</b>
<b>CUDA code starts here</b>
</pre>
