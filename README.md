# Image-Captioning-With-Attention
Tensorflow implementation of attention mechanisms for image captioning


## Access H-BRS cluster

Open a terminal to access the cluster

```
ssh user2s@wr0.wr.inf.h-brs.de

```

user2s is your own student identifier. After that type your cluster password and you'll be logged in the cluser.

## Install Anaconda in H-BRS cluster

Now create a temporal directory to store the anaconda installation file. In your cluster home directory do 

```
mkdir temporal

```

Then proceed with the Anaconda installation in your cluster account. [Install Anaconda in Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart).

## Copy python scripts to H-BRS cluster

In your cluster home folder, create a main directory to store the python scripts

```
mkdir image_captioning

```

In terminal type the following

```
scp /home/hackerman/Documents/Alan-Git-Repositories/Natural-Language-Processing/Project/model.py apreci2s@wr0.wr.inf.h-brs.de:/home/apreci2s/Image_captioning

```



