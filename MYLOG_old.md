## 07/26/2021
- [Shell字符串详解](http://c.biancheng.net/view/821.html)
  - <img width="1017" alt="Screen Shot 2021-07-26 at 4 40 03 PM" src="https://user-images.githubusercontent.com/32129905/127056053-65f10233-443b-4b8f-8137-a31b31301987.png">


## 07/24/2021
- [Shell字符串拼接（连接、合并）](http://c.biancheng.net/view/1114.html)
  - <img width="631" alt="Screen Shot 2021-07-26 at 4 38 06 PM" src="https://user-images.githubusercontent.com/32129905/127055835-7de4cbc3-a29c-4e2d-9b21-fa909d0f4705.png">
- [一篇教会你写90%的shell脚本](https://zhuanlan.zhihu.com/p/264346586)
- [curl shell](https://github.com/why2011btv/Quizlet_6/blob/master/te_out/my.sh)

## 07/23/2021
- Why does "source venv/bin/activate" not work?
  - <img width="1178" alt="Screen Shot 2021-07-23 at 12 52 02 AM" src="https://user-images.githubusercontent.com/32129905/126738758-6257a0b5-00b9-4fbf-990c-be1bc10468bf.png">
  - Tentative solution ([reference](https://www.devdungeon.com/content/python-import-syspath-and-pythonpath-tutorial#toc-5)): 
    - ```export VIRTUAL_ENV=/shared/public/ben/temporalextraction/venv```
    - ```export PATH="$VIRTUAL_ENV/bin:$PATH"```
    - ```export PYTHONHOME=/shared/public/ben/temporalextraction/venv```
    - ```export PYTHONPATH=/shared/public/ben/temporalextraction/venv/lib/python3.6```
  - [A similar problem](https://github.com/ContinuumIO/anaconda-issues/issues/172)
  - [Is my virtual environment (python) causing my PYTHONPATH to break?](https://stackoverflow.com/questions/4686463/is-my-virtual-environment-python-causing-my-pythonpath-to-break)
  - [How do you set your pythonpath in an already-created virtualenv?](https://stackoverflow.com/questions/4757178/how-do-you-set-your-pythonpath-in-an-already-created-virtualenv)
  - ![Screen Shot 2021-07-24 at 12 09 50 AM](https://user-images.githubusercontent.com/32129905/126856867-07252935-5240-47ed-952d-648b6c9505da.png)
  - [python编程【环境篇】- 如何优雅的管理python的版本](https://www.cnblogs.com/zhouliweiblog/p/11497045.html)
  - [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)
- [Quizlet 6 Draft Pipeline](https://github.com/why2011btv/Quizlet_6)



<html lang="en-US">
    <meta charset="UTF-8">
<body>

<h1>MYLOG_old</h1>
<h2>07/23/2020</h2>
<h3>conda activate /shared/why16gzl/logic_driven/consistency_env</h3>
<h3>conda activate myenv</h3>
<h3>source ~/venv/bin/activate</h3>
<h3>source /shared/why16gzl/logic_driven/mandarjoshi90_coref/mandarjoshi90_coref_env/bin/activate</h3>
<li><p><a href="https://www.w3schools.com/python/gloss_python_global_variables.asp">Python Global Variables</a>: "global x" inside a function does not define a new variable</p></li>
<li><p><a href="https://www.geeksforgeeks.org/multiprocessing-python-set-2/">Multiprocessing in Python</a>: any newly created process will have their own memory space</p></li>
<li><p><a href="https://towardsdatascience.com/virtual-environments-104c62d48c54">Creating a Virtual Environment</a>:<br>mkdir new_project<br>cd new_project<br>python3 -m venv venv&#60;name_of_virtualenv&#62;<br>source venv/bin/activate<br>deactivate<br>pip freeze &#62; requirements.txt</p></li>
<li><p><a href="https://realpython.com/python-virtual-environments-a-primer/">Distinguishing: virtualenv, virtualenvwrapper, and pyenv</a></p></li>
<li><p><a href="https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533">Distinguishing: conda, pip, and venv</a>:<br>conda create --prefix /path/to/conda-env<br>conda create --name conda-env python #installed in /home1/w/why16gzl/miniconda3/envs/</p></li>
Given an environment.yml file, you can easily recreate an environment.<br>
% conda env create -n conda-env -f /path/to/environment.yml<br>
Bonus: You can also add the packages listed in an environment.yml file to an existing environment with:<br>
% conda env update -n conda-env -f /path/to/environment.yml
<li><p><a href="https://realpython.com/python-virtual-environments-a-primer/">virtualenvwrapper</a></p></li>
<h2>07/24/2020</h2>
<h3>vi line navigation</h3>
0: go to start of line<br>
$: go to end of line<br>
1G: go to top of file<br>
5G: go to line 5<br>
G: go to end of file<br>
/: search for string<br>
    
    
    
<h2>07/25/2020</h2>
<h3>set up environment for <a href="https://github.com/mandarjoshi90/coref/">SpanBERT coref part</a></h3>
pip install cort:<br>git clone https://github.com/smartschat/cort.git<br>go to setup.py line 37, delete "mmh3"<br>conda install mmh3<br>python setup.py install
<h3><a href="https://linuxize.com/post/how-to-create-symbolic-links-in-linux-using-the-ln-command/">Create Symbolic Links</a></h3>
ln -s /shared/why16gzl/miniconda3/myenv/ /home1/w/why16gzl/miniconda3/envs/myenv<br>
conda activate myenv (or consistency_env, coref_SpanBERT_TF_env, directly without specifying whole path)
<h3>ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory</h3>
See /home1/w/why16gzl/0_markdown/TF_ipykernel_Configuration.ipynb<br>
Now myenv has tensorflow==1.12.0
<h3>List the size of directory: du -h --max-depth=1 | sort -n</h3>
<h2>07/26/2020</h2>
<h3>Improving Implicit Argument and Explicit Event Connections for Script Event Prediction, EMNLP 2020 Submission</h3>
To capture explicit evolutionary relationships between events, we follow (Li et al., 2018) to construct an event graph from intersecting chains, and further develop graph convolutional layers based on GCN (Kipf and Welling, 2017) to extract fine-grained features.
<h3>Constructing Narrative Event Evolutionary Graph for Script Event Prediction, IJCAI 2018</h3>
In order to compare with previous work, we adopt the same news corpus and event chains extraction methods as [Granroth-Wilding and Clark, 2016].
<h3>What Happens Next? Event Prediction Using a Compositional Neural Network Model, AAAI 2016</h3>
This paper follows a line of work begun by Chambers and Jurafsky (2008), who introduced a technique for automatically extracting knowledge about typical sequences of events from text.<br>
we propose a new task, multiple choice narrative cloze (MCNC)<br>
The event extraction pipeline follows an almost identical procedure to Chambers and Jurafsky (2009), using the C&amp;C tools (Curran, Clark, and Bos 2007) for PoS tagging and dependency parsing and OpenNLP for phrase-structure parsing and coreference resolution.
<h3>Unsupervised learning of narrative event chains, ACL 2008</h3>
Narrative Cloze Task
<h2>07/27/2020</h2>
<h3>How to run OneIE</h3>
conda activate consistency_env<br>
python predict.py -m ../english.role.v0.3.mdl -i input -o output -c output_cs --format ltf
<h3>How to run SpanBERT</h3>
conda activate coref_SpanBERT_TF_env<br>
CUDA_VISIBLE_DEVICES=0 python predict.py spanbert_large cased_config_vocab/trial.jsonlines 0725.out
<h3>How to build the event graph</h3>
First run OneIE to get: segment-15 ... [3, 5, "GPE", "NAM", 1.0] ... [9, 10, "Contact:Meet", 1.0] ... [1, 0, "ORG-AFF", 0.52] ... [0, 2, "Attacker", 0.5514472874407822]. Represent each entity/event as their #segment_#start_token_index_#end_token_index: 15_3_5
<br>
Then run SpanBERT. SpanBERT input can be several sentences. But the token num is limited.
<h2>08/05/2020</h2>
<h3>How to debug</h3>
<h3><a href="https://www.codementor.io/@stevek/advanced-python-debugging-with-pdb-g56gvmpfa">Post-mortem debugging</a></h3>
python3 -mpdb script.py<br>
To start execution, you use the continue or c command. If the program executes successfully, you will be taken back to the (Pdb) prompt where you can restart the execution again. If the program throws an unhandled exception, you'll also see a (Pdb) prompt, but with the program execution stopped at the line that threw the exception. From here, you can run Python code and debugger commands at the prompt to inspect the current program state.
<h3><a href="https://realpython.com/lessons/continuing-execution/">A tutorial on pdb</a></h3>
<h3><a href="http://qingkaikong.blogspot.com/2018/05/python-debug-in-jupyter-notebook.html">Python debug in Jupyter notebook</a></h3>
%debug
<h3><a href="https://davidhamann.de/2017/04/22/debugging-jupyter-notebooks/">Debugging Jupyter notebooks</a></h3>
<h3><a href="https://stackoverflow.com/questions/2534480/proper-way-to-reload-a-python-module-from-the-console">Proper way to reload a python module from the console</a></h3>
import document_reader<br>
import importlib<br>
importlib.reload(document_reader)<br>
from document_reader import *<br>
<h3><a href="https://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing">Unsolved Problem: nltk.tokenizer would change "" to ``''</a></h3>
<h3><a href="https://mccormickml.com/2019/07/22/BERT-fine-tuning/#4-train-our-classification-model">BERT Fine-Tuning Tutorial with PyTorch</a></h3>
<h3><a href="">xxx</a></h3>
<h2>08/16/2020</h2>
<h3>Install CUDA 10.2</h3>
1) go to <a href="https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=OpenSUSE&target_version=15&target_type=runfilelocal">nvidia website</a><br>
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run<br>
chmod a+x cuda_10.2.89_440.33.01_linux.run<br>
sh ./cuda_10.2.89_440.33.01_linux.run<br>
2) change the options (installation path) <a href="https://stackoverflow.com/questions/39379792/install-cuda-without-root">see answer by fr_andres</a><br>
3) add below to .bashrc<br>
#cuda-10.2<br>
export PATH=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2/bin:$PATH<br>
export LD_LIBRARY_PATH=/shared/why16gzl/Downloads/Downloads/cuda_10_2/cuda-10.2/lib64<br>
<h3><a href="https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory">How do I list all files of a directory</a></h3>
<h3><a href="https://github.com/thunlp/DocRED">Dataset and code for baselines for DocRED: A Large-Scale Document-Level Relation Extraction Dataset</a></h3>
<a href="https://www.aclweb.org/anthology/P08-1090.pdf">Unsupervised Learning of Narrative Event Chains</a>
<h3><a href="http://qingkaikong.blogspot.com/2018/05/python-debug-in-jupyter-notebook.html"> Python debug in Jupyter notebook </a></h3>
<h2>08/17/2020</h2>
<h3>torch.stack</h3>
In [ ]:<br>
a = torch.tensor([0, 1, 2])<br>
b = []<br>
b.append(a)<br>
b.append(a)<br>
torch.stack(b)<br>
Out [ ]:<br>
tensor([[0, 1, 2],<br>
        [0, 1, 2]])<br>
<h3>RoBERTa token id</h3>
&#60;s&#62;: 0<br>
&#60;pad&#62;: 1<br>
&#60;/s&#62;: 2<br>
&#60;unk&#62;: 3<br>
<h3><a href="https://www.teachucomp.com/special-characters-in-html-tutorial/">special characters in html</a></h3>
<h3>How to run my model (entity coref as context)</h3>
/shared/why16gzl/logic_driven/Untitled1.ipynb<br>
coref_flag = True: train_set_coref is generated<br>
coref_flag = False: train_set is generated<br>
Under folder /shared/why16gzl/logic_driven/data_pkl
<h2>08/20/2020</h2>
<h3>Update github repo</h3>
git add .<br>
git commit -m "add googlescholar"<br>
git config http.postBuffer 524288000 # if the next line does not work you might try this line first<br>
git push origin master<br>
<h3><a href="https://www.aclweb.org/anthology/W18-4702.pdf">Interoperable Annotation of Events and Event Relations across Domains</a></h3>
<h2>08/25/2020</h2>
<h3>Not using pickle, training after dataloader</h3>
(consistency_env) why16gzl@morrison:/shared/why16gzl/logic_driven/EMNLP_2020&#62; nohup python3 main_0824.py gpu_3 batch_28 0.00001 0824_1.rst epoch_60 &#62; 0824_1.out 2&#62;&#38;1 &#38;<br>
<h3>Using pickle, coref or no_coref</h3>
<h4>Preprocessing, Coref</h4>
(allennlp_env) why16gzl@morrison:/shared/why16gzl/logic_driven&#62; python why16gzl_prep.py 1
<h4>Preprocessing, No Coref</h4>
(allennlp_env) why16gzl@morrison:/shared/why16gzl/logic_driven&#62; python why16gzl_prep.py 0
<h4>Training</h4>
(consistency_env) why16gzl@morrison:/shared/why16gzl/logic_driven&#62; nohup python3 main.py gpu_1 batch_24 no_coref 0817_0.rst epoch_60 &#62; 0817_0.out 2&#62;&#38;1 &#38;<br>
<h2>09/05/2020</h2>
<h3>CogCompNLP environment</h3>
A JAR (Java ARchive) is a package file format typically used to aggregate many Java class files and associated metadata and resources (text, images, etc.) into one file for distribution. JAR files are archive files that include a Java-specific manifest file.<br>
Maven is a build automation tool used primarily for Java projects. Maven can also be used to build and manage projects written in C#, Ruby, Scala, and other languages. The Maven project is hosted by the Apache Software Foundation, where it was formerly part of the Jakarta Project.<br>
<h2>09/20/2020</h2>
Create a new repository on GitHub. You can also add a gitignore file, a readme and a licence if you want<br>
 Open Git Bash<br>
Change the current working directory to your local project.<br>
Initialize the local directory as a Git repository.<br>
git init<br>
Add the files in your new local repository. This stages them for the first commit.<br>
git add .<br>
 Commit the files that you’ve staged in your local repository.<br>
git commit -m "initial commit"<br>
 Copy the https url of your newly created repo<br>
In the Command prompt, add the URL for the remote repository where your local repository will be pushed.<br>

git remote add origin remote repository URL<br>

git remote -v<br>
 Push the changes in your local repository to GitHub.<br>

git push -f origin master<br>
https://www.softwarelab.it/2018/10/12/adding-an-existing-project-to-github-using-the-command-line/<br>
https://stackoverflow.com/questions/15244644/how-to-restart-a-git-repository<br>
https://stackoverflow.com/questions/32238616/git-push-fatal-origin-does-not-appear-to-be-a-git-repository-fatal-could-n<br>
<h2>09/22/2020</h2>
https://shmsw25.github.io<br>
https://seominjoon.github.io<br>
https://luheng.github.io/files/cv_luheng_2017.pdf<br>

HW1:<br>
Part 1: Querying an AWS Oracle database (IMDB database)<br>
Use "Oracle SQL Developer" to connect to IMDB database<br>
Part 2: Setting up and querying an OpenFlights dataset using MySQL (MariaDB)<br>
<h2>09/28/2020</h2>
"pip install en-core-web-sm==2.1.0" fails, instead, do the following:<br>
python -m spacy download en_core_web_sm<br>

Go to end of file (vi): &#60;ESC&#62;GA

<h2>10/04/2020</h2>
linux 下后台运行python脚本
<br>
https://www.jianshu.com/p/4041c4e6e1b0


<h2>01/09/2021</h2>

$ git tag -a v1.4 -m "my version 1.4"
<br>
https://git-scm.com/book/en/v2/Git-Basics-Tagging
<br>
git add . <br>
git commit -m "xxx" <br>
git push origin v1.4
<br>

<h2>01/10/2021</h2>
multiprocessing
<br>
https://pymotw.com/2/multiprocessing/basics.html
<br>
threading
<br>
https://pymotw.com/2/threading/index.html#module-threading
<br>

<h2>01/18/2021</h2>
git checkout -b <branch> <br>
Edit files, add and commit. Then push with the -u (short for --set-upstream) option:
<br>
git push -u origin <branch>
<br>

https://stackoverflow.com/questions/2765421/how-do-i-push-a-new-local-branch-to-a-remote-git-repository-and-track-it-too

<h2>03/19/2021</h2>
https://www.bradyneal.com/which-causal-inference-book#elements-of-causal-inference<br>
<h2>03/21/2021</h2>

import notify<br>
see: https://github.com/why2011btv/JCL_EMNLP20/blob/main/notify_message.py and https://github.com/why2011btv/JCL_EMNLP20/blob/main/notify_smtp.py and https://github.com/why2011btv/JCL_EMNLP20/blob/0ede3a5c205c57b6e82499ad5ed46f2af63acafc/exp.py#L250<br>



shift+G: go to end of file (vi)<br>
How to remove files already added to git after you update .gitignore: git rm -r --cached .<br>
https://itnext.io/how-to-remove-files-already-added-to-git-after-you-update-gitignore-90f169a0a4e1<br>
https://www.javatpoint.com/git-head<br>
To delete the last line of file: sed -i '$d' FILENAME<br>
git ls-files<br>

https://opensource.com/article/18/6/git-reset-revert-rebase-commands<br>
$ git log --oneline<br>
b764644 File with three lines<br>
7c709f0 File with two lines<br>
9ef9173 File with one line<br>

$ git reset 9ef9173 (using an absolute commit SHA1 value 9ef9173)<br>
<h2>03/22/2021</h2>
http://www.filepermissions.com/directory-permission/2750<br>
<h2>03/31/2021</h2>
How to get the root node: in-degree == 0<br>
https://stackoverflow.com/questions/4122390/getting-the-root-head-of-a-digraph-in-networkx-python<br>

<h2>04/24/2021</h2>
git push to github, not showing user picture in commit info, but only "Haoyu Wang": <br>
https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address<br>
https://stackoverflow.com/questions/48816950/showing-gitlab-profile-picture-on-commits<br>
You need to: git config user.email "your email here"<br>
https://docs.github.com/en/github/setting-up-and-managing-your-github-profile/why-are-my-contributions-not-showing-up-on-my-profile#commit-was-made-less-than-24-hours-ago<br>
Why sometimes it does not count towards your contribution?


</body>
</html>
    
## 07/12/2021
<img width="1361" alt="Screen Shot 2021-07-12 at 11 56 23 PM" src="https://user-images.githubusercontent.com/32129905/125388101-e30a2c80-e36c-11eb-84ad-8ac0053ba2fc.png">
    
## 07/13/2021
pip list
    
