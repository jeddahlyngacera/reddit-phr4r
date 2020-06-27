
# Reddit, Set, Go: Deciphering What Redditors Are Really Looking For
## *What are Filipino Redditors really looking for in dating in r/phr4r*

**Big Data and Cloud Computing Final Project**

**MSDS 2020 Learning Team 4**
 * Ria Ysabelle L. Flora
 * Crisanto E. Chua
 * Armand Louis A. De Leon
 * Jeddahlyn V. Gacera
 
**Asian Institute of Management**

# I. Executive Summary

Given the privilege of `anonymity`, `Reddit` has always been a platform that enabled its users to `explicitly post` about what they want, with special subreddits being created for several niche topics. Subreddits catering to these are often tagged as `r4r` or `redditor-for-redditor` and with that, the aim of this study is to extract all the information on the posts from the `Philippine-based subreddit r/phr4r` to paint a clearer picture of the `dating` and `hook-up culture` in the Philippines. Collection of the data was done through an updated source in Jojie. `Frequent Itemset Mining (FIM)` and `Association Rule Mining (ARM)` was employed to extract the most frequently occurring verbs, adjectives, and nouns from individual posts. Data points were identified according to `gender`, `age`, and `sexual orientation` due to the consistent topic name formatting imposed in the subreddit. The demographic information combined with the FIM and ARM output was used for the profiling of different age groups or sexual orientation to create a more granular analysis.

# II. Introduction

`Reddit` is an online messaging board established at the University of Virginia in 2005 and is currently, according to its website, the `5th most visited site` in the United States of America (USA). Users at this site are distinguished by their unique user identifications (user ID) whereas the content posted in the website are user-generated and vary in types: text, video, and links. These contents are further organized according to user-generated communities called `subreddits`, or subs as popularly called, which cater to different arrays of interests – encompassing a wide range of niche and general topics. Each subreddit has its own theme, culture, and rules (as set by its moderators). 

One of the popular subreddits would be `r/r4r` which stands for `redditor – for – redditor` wherein community users, post personal to meet other users. Posts in this subreddit cover dating, marriage proposals, academic support, video games, poetry, playing board games, and many more. Since its cake day in April 2010, it currently has a registry of 284,000 users. International versions of which have also emerged, such as `r/phr4r in the Philippines`, `r/euro4euro for Europe`, `r/r4rtoronto for Toronto Canada`, and many more; all of which, are anchored on the same objective of connecting Reddit users to each other for whichever purpose indicated. Nonetheless, a distinct feature across these subreddits would be the formatting at which its posts are written accordingly: &lt;age&gt; [&lt;r4r&gt;] &lt;title&gt; wherein users fill in the tags in between the angled brackets with r being indicative of the user’s gender and gender preference.  

Aside from the distinct text formatting in these subreddits, r4r subreddits are distinct for the `cloak of anonymity` it provides to its users which is disparate from other dating and social media platforms.  It is also worth noting that all posts in these subreddits are purely textual and do not include images – and thus establishes complete anonymity for each post. Given this, and in line with a study by (Curlew, 2019), forms social media platforms that subscribe to real name basis lead to identity curation whereas platforms that enable its users to be anonymous through usernames and/or complete anonymity lead to less inhibitions and more elaborate forms of self-expression. Hence, through Reddit’s r4r subreddits, users are provided with an `avenue to express themselves without inhibitions brought by social factors` such as family dynamics, public reputation, and body image.    

Given the growing number of Reddit users and the blanket of anonymity it provides, the objective of the study is to find out `what Filipino Redditors are looking for in dating in the Subreddit r/phr4r`. Accomplishing this would entail extracting information on the posts through `data mining techniques and natural language processing` – all towards painting a clearer picture of the `dating and hook-up culture in the Philippines`. This study is particularly anchored on the nature of the subreddit r/phr4r wherein majority of the posts are particularly on dating and hook ups with hook ups being defined as a casual encounter ranging from kissing to intercourse with no defined expectations on progressing the relationship towards a fully committed one (Black, S., et al., 2019).  

The study shall be delimited to the r/phr4r subreddit and shall not encompass any other similar subreddits of the same nature. Moreover, the study only took into accounts posted from January 2016 to August 2019.

# III. Methodology

The following steps were taken for this study:

**1. Data Description and Collection**<br>
* Data gathering<br>
* Text data<br>

**2. Data Cleaning and Preprocessing**<br>
* POS tagging<br>
* N-grams<br>

**3. Exploratory Data Analysis**<br>

**4. Profiling**

**5. Data Mining**<br>

* Frequent Itemset Mining<br>
* Association Rule Mining

## 1.	Data Description and Collection

Data on this study is from one of the most popular social platforms, `Reddit`, wherein users can explicitly post about what they want, with special subreddits being created for several niche topics. Specifically, we will be analyzing the Philippine-based subreddit `r/phr4r` under the main subreddit r/r4r.

Due to its popularity and robustness, it is one of the websites being collected for big data analysis. <a href="https://files.pushshift.io/">Pushshift.io</a> is a site maintained by Jason Baumgartner and contains various articles relating to big data, social media ingest and analysis and general technology trends. This includes Reddit data (redditors, subreddits, submittions, comments, etc.) which can be found <a href="https://files.pushshift.io/reddit/submissions/">here</a>.

### a. Data gathering

All reddit submissions data found in Pushshift.io were copied to jojie (`/mnt/data/public/reddit/submissions/RS_<YYYY>_<MM>.*`). These `46 files` were then copied to an `s3 bucket`, and then to an `efs` mounted to all `AWS instances` (master and slaves) for processing.


```python
from dask.distributed import Client
client = Client('54.249.93.186:8786')
client
```




<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://54.249.93.186:8786</li>
  <li><b>Dashboard: </b><a href='http://54.249.93.186:8080/status' target='_blank'>http://54.249.93.186:8080/status</a>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Cluster</h3>
<ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
  <li><b>Workers: </b>12</li>
  <li><b>Cores: </b>24</li>
  <li><b>Memory: </b>98.68 GB</li>
</ul>
</td>
</tr>
</table>




```python
import dask
import dask.dataframe as dd
import dask.bag as db
import pandas as pd
import pickle
import glob
import json
import re
```

#### Selected features


```python
features = ['subreddit', 'created_utc', 'author', 'title', 
            'selftext', 'ups', 'downs', 'score', 'num_comments']
```


```python
def get_features(r):
    '''Returns a dictionary with features as keys and values from the data 
       found in file for that key'''
    j = json.loads(r)
    return json.dumps([j[f] if f in j else None for f in features])
```

#### Select phr4r subreddit


```python
sub = re.compile(r'"subreddit":"phr4r"')

def re_filter(r):
    '''Returns True if the file containg phr4r, else False'''
    if sub.search(r):
        return True
    return False
```

#### Process bz2 files (January 2016 - October 2017) and save to a pickle file


```python
bz_raw = db.read_text('/home/ubuntu/efs/submissions/RS*.bz2')
```


```python
bz_sub = bz_raw.filter(re_filter).map(get_features)
```


```python
bz_sub.to_textfiles(path='/home/ubuntu/efs/bz_files/')
```




    ['/home/ubuntu/efs/bz_files/00.part',
     '/home/ubuntu/efs/bz_files/01.part',
     '/home/ubuntu/efs/bz_files/02.part',
     '/home/ubuntu/efs/bz_files/03.part',
     '/home/ubuntu/efs/bz_files/04.part',
     '/home/ubuntu/efs/bz_files/05.part',
     '/home/ubuntu/efs/bz_files/06.part',
     '/home/ubuntu/efs/bz_files/07.part',
     '/home/ubuntu/efs/bz_files/08.part',
     '/home/ubuntu/efs/bz_files/09.part',
     '/home/ubuntu/efs/bz_files/10.part',
     '/home/ubuntu/efs/bz_files/11.part',
     '/home/ubuntu/efs/bz_files/12.part',
     '/home/ubuntu/efs/bz_files/13.part',
     '/home/ubuntu/efs/bz_files/14.part',
     '/home/ubuntu/efs/bz_files/15.part',
     '/home/ubuntu/efs/bz_files/16.part',
     '/home/ubuntu/efs/bz_files/17.part',
     '/home/ubuntu/efs/bz_files/18.part',
     '/home/ubuntu/efs/bz_files/19.part',
     '/home/ubuntu/efs/bz_files/20.part',
     '/home/ubuntu/efs/bz_files/21.part']




```python
df_bz = dd.read_json('/home/ubuntu/efs/bz_files/*', lines=True).compute()
```


```python
df_bz.shape
```




    (5494, 9)




```python
df_bz.columns = features
```


```python
df_bz['texts'] = df_bz['title']+'\n'+df_bz['selftext']
```


```python
df_bz.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phr4r</td>
      <td>1451891616</td>
      <td>EdWao</td>
      <td>[m4m] looking for FUN</td>
      <td>Hey. I'm 18 bi m from Sta. Mesa Manila. Lookin...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>8</td>
      <td>[m4m] looking for FUN\nHey. I'm 18 bi m from S...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phr4r</td>
      <td>1451950820</td>
      <td>[deleted]</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]</td>
      <td>[removed]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phr4r</td>
      <td>1452053372</td>
      <td>[deleted]</td>
      <td>30 [M4F] Shaw, Movie and chill</td>
      <td>[deleted]</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>30 [M4F] Shaw, Movie and chill\n[deleted]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phr4r</td>
      <td>1452065137</td>
      <td>ThreesomePH</td>
      <td>23 [M4F] Manila area. Actually looking for a t...</td>
      <td>So my girlfriend and I want to try something n...</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>8</td>
      <td>23 [M4F] Manila area. Actually looking for a t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phr4r</td>
      <td>1452248658</td>
      <td>[deleted]</td>
      <td>30 [M4F] San Pablo Laguna, hangout?</td>
      <td>[deleted]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>30 [M4F] San Pablo Laguna, hangout?\n[deleted]</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open('/home/ubuntu/efs/files/df_bz.pkl', 'wb') as f:
    pickle.dump(df_bz, f)
```

#### Process xz files (November 2017- October 2018) and save to a pickle file


```python
xz_raw = db.read_text('/home/ubuntu/efs/submissions/RS*.xz')
```


```python
xz_sub = xz_raw.filter(re_filter).map(get_features)
```


```python
xz_sub.to_textfiles(path='/home/ubuntu/efs/xz_files/')
```




    ['/home/ubuntu/efs/xz_files/00.part',
     '/home/ubuntu/efs/xz_files/01.part',
     '/home/ubuntu/efs/xz_files/02.part',
     '/home/ubuntu/efs/xz_files/03.part',
     '/home/ubuntu/efs/xz_files/04.part',
     '/home/ubuntu/efs/xz_files/05.part',
     '/home/ubuntu/efs/xz_files/06.part',
     '/home/ubuntu/efs/xz_files/07.part',
     '/home/ubuntu/efs/xz_files/08.part',
     '/home/ubuntu/efs/xz_files/09.part',
     '/home/ubuntu/efs/xz_files/10.part',
     '/home/ubuntu/efs/xz_files/11.part']




```python
df_xz = dd.read_json('/home/ubuntu/efs/xz_files/*', lines=True).compute()
```


```python
df_xz.shape
```




    (13711, 9)




```python
df_xz.columns = features
```


```python
df_xz['texts'] = df_xz['title']+'\n'+df_xz['selftext']
```


```python
df_xz.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phr4r</td>
      <td>1509505274</td>
      <td>watchmeplay_</td>
      <td>25[M4F] Wanna watch?</td>
      <td>[removed]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>25[M4F] Wanna watch?\n[removed]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phr4r</td>
      <td>1509507757</td>
      <td>123Hihellobye</td>
      <td>25 [F4M] looking for a friend with a big heart 😉</td>
      <td>Just kidding 😀\n\nLooking for a good friend wh...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>12</td>
      <td>25 [F4M] looking for a friend with a big heart...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phr4r</td>
      <td>1509509229</td>
      <td>[deleted]</td>
      <td>32 [M4F] Manila looking for conversation</td>
      <td>[deleted]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>32 [M4F] Manila looking for conversation\n[del...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phr4r</td>
      <td>1509511793</td>
      <td>[deleted]</td>
      <td>25 [M4F] NSA/FWB</td>
      <td>[deleted]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>25 [M4F] NSA/FWB\n[deleted]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phr4r</td>
      <td>1509512861</td>
      <td>[deleted]</td>
      <td>F4R Pasig Area.</td>
      <td>[removed]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>F4R Pasig Area.\n[removed]</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open('/home/ubuntu/efs/files/df_xz.pkl', 'wb') as f:
    pickle.dump(df_xz, f)
```

#### Process zst files (November 2018- August 2019) and save to a pickle file


```python
zst_raw = db.read_text('/home/ubuntu/efs/submissions/RS*.zst')
```


```python
zst_sub = zst_raw.filter(re_filter).map(get_features)
```


```python
zst_sub.to_textfiles(path='/home/ubuntu/efs/zst_files/')
```




    ['/home/ubuntu/efs/zst_files/0.part',
     '/home/ubuntu/efs/zst_files/1.part',
     '/home/ubuntu/efs/zst_files/2.part',
     '/home/ubuntu/efs/zst_files/3.part',
     '/home/ubuntu/efs/zst_files/4.part',
     '/home/ubuntu/efs/zst_files/5.part',
     '/home/ubuntu/efs/zst_files/6.part',
     '/home/ubuntu/efs/zst_files/7.part',
     '/home/ubuntu/efs/zst_files/8.part',
     '/home/ubuntu/efs/zst_files/9.part']




```python
df_zst = dd.read_json('/home/ubuntu/efs/zst_files/*', lines=True).compute()
```


```python
df_zst.shape
```




    (42474, 9)




```python
df_zst.columns = features
```


```python
df_zst['texts'] = df_zst['title']+'\n'+df_zst['selftext']
```


```python
df_zst.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phr4r</td>
      <td>1541030869</td>
      <td>Jsadf313d1</td>
      <td>23[M4F] South. Let’s get rid of stress over th...</td>
      <td>[removed]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>23[M4F] South. Let’s get rid of stress over th...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phr4r</td>
      <td>1541030922</td>
      <td>craigchina</td>
      <td>29/24 MF4F - travel guide and more</td>
      <td>[removed]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>29/24 MF4F - travel guide and more\n[removed]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phr4r</td>
      <td>1541031135</td>
      <td>Pattaptap</td>
      <td>MF4M looking for a thirdy cavite area.</td>
      <td>[removed]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>MF4M looking for a thirdy cavite area.\n[removed]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phr4r</td>
      <td>1541031626</td>
      <td>MoreAboutChances</td>
      <td>22 [M4F] UPD - Sexual frustrations in college ...</td>
      <td>Hello there. Sa totoo lang sobrang stress ko t...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>22 [M4F] UPD - Sexual frustrations in college ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phr4r</td>
      <td>1541035382</td>
      <td>Pluckedwings</td>
      <td>[28F4A] Anyone game to hangout this long weeke...</td>
      <td>[removed]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>[28F4A] Anyone game to hangout this long weeke...</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open('/home/ubuntu/efs/files/df_zst.pkl', 'wb') as f:
    pickle.dump(df_zst, f)
```

#### Combine all and prepare for data cleaning using NLP


```python
df_all = df_bz.append(df_xz).append(df_zst)[['texts']]
df_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[m4m] looking for FUN\nHey. I'm 18 bi m from S...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30 [M4F] Shaw, Movie and chill\n[deleted]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23 [M4F] Manila area. Actually looking for a t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30 [M4F] San Pablo Laguna, hangout?\n[deleted]</td>
    </tr>
  </tbody>
</table>
</div>



#### Save to a pickle file


```python
with open('/home/ubuntu/efs/files/df_all.pkl', 'wb') as f:
    pickle.dump(df_all, f)
```

### b. Text data

It is possible to mine the age and sexual preference of phr4r users due to the standard format that each post must follow in accordance with the subreddit’s rules. Each post must have the following format:  

#### <center>&lt;AGE&gt; [(M/F)4(M/F/A)] &lt;Post title/header&gt;</center>
 

For example, a straight, 24-year-old male will indicate this as “24 [M4F]”, while a homosexual 22-year-old female will use “22 [F4F]”. The “A” is used to indicate that the poster is indifferent to the sex of the person they are looking for. Posts by couples may also be formatted as [MF4A] or [FF4M], indicating a couple is interested in a third party. These posts are filtered out from scraping to keep analysis simple. 

 

The bodies of these posts vary in content, although a common format used in most posts is as follow: 

<pre> 
An introduction to provide context

“About me”, which enumerates enough traits and attributes to describe but not give away identity 

“About you”, enumerates the user’s requirements or preferences for the person they would like to meet 

</pre>

Due to most posts having descriptions of themselves and preferred other, it would make sense to filter the text according to only nouns, verbs, adjectives, and adverbs. The process of tagging and filtering according to part of speech is detailed later in this paper. 

## 2. Data Cleaning and Preprocessing
Text pre-processing is the first step in preparing text data for analysis. For this study, the following were done to clean the text before part of speech (POS) tagging: 

1. Lowercase all characters 

2. Remove all special characters and punctuations 

3. Lemmatizing and POS tagging (done simultaneously) 

4. Stop word removal 

As an additional note, stop words are removed after POS tagging since the model involved in POS tagging may lose information it needs to properly tag words.

### a. POS tagging 

POS tagging allows a machine to properly identify how a certain word was used within a sentence, whether it was used as a noun, adjective, verb, etc. This has multiple applications, such as in telling a machine how to pronounce a word in text-to-speech (TTS) programs (Bellegarda, 2015), in aspect-level sentiment analysis, or automated grammar checking. POS tagging makes it possible to filter out particular parts of speech that may not be relevant to the analysis, and in a way can be used as a crude means of dimensionality reduction. 

For this study, the POS tagger in the SpaCy library was used. Hence, lemmatizing is done alongside the POS tagging process. Parts of speech such as nouns, verbs, adjectives, and adverbs were retained for analysis.

### b. N-grams 

N-grams were not used since it may conflict with the point of the study, which is to find which words are most likely to occur together. It may be redundant to consider two-word sequences since the concern of the study is to find relationships between single words in the posts. 

Sexual Orientation 

As mentioned earlier, the format <AGE> [(M/F)4(M/F/A)] <Post title/header> allows us to infer the sexual orientation of phr4r users. For simplicity, posts seeking the opposite sex are tagged straight and posts seeking the same sex are tagged LGBT. Posts such as [M4A] or [F4A] are tagged as indeterminate and filtered out of the analysis.


```python
import pickle
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('brown')
nltk.download('universal_tagset')
import string 
import gensim
from gensim import corpora

nlp = spacy.load('en_core_web_sm')
stops2 = spacy.lang.en.stop_words.STOP_WORDS

word_lem = WordNetLemmatizer()
```

#### Load combined dataframe


```python
with open('df_all.pkl', 'rb') as f:
    df_all = pickle.load(f)
```


```python
df_all.reset_index(drop=True, inplace=True)
```


```python
df_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>[m4m] looking for FUN\nHey. I'm 18 bi m from S...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>30 [M4F] Shaw, Movie and chill\n[deleted]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>23 [M4F] Manila area. Actually looking for a t...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>30 [M4F] San Pablo Laguna, hangout?\n[deleted]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_all.loc[10, 'texts']
```




    "22 [M4F], Vito Cruz-DLSU area. In a relationship, trying to find someone on the same bench.\nSomething something casual and fun. Haha. Sorry for the boring description. Just PM and we'll get started.\n\nCheers!"



#### Text Super Pre-Processor


```python
stops = stopwords.words('english') + [
            '-PRON-', 'someone', 'somebody', 'lf', 'look', 'go', 'get', 'let', 
           'want', 'know', 'sa', 'ang', 'ng', 'nang', 'mo', 'ni', 'ka', 'kay', 
           'at', 'ko', 'kasi', 'pero', 'wanna', 'para', 'lang', 'ig', 'na', 
           'ako', 'naman', 'tapos', 'di', 'din', 'rin', 'ang', 'kung', 'siya', 
           'ig', 'im', 'na', 'mga', 'uhmm', 'uhm', 'to', 'hahaha', 'yung', 
           'haha', 'basta', 'f', 'baka', 'walang', 'akong', 'kasama', 'tara', 
           'dito', 'ba', 'pwede', 'lol', 'ano', 'po', 'sana', 'pag', 'g', 
           'tayo', 'talaga', 'female', 'guy', 'hindi', 'kaya', 'pa', 'medyo', 
           'usap', 'ps', 'inom', 'mag', 'man', 'kahit', 'gusto', 'wala', 
           'amp', 'give', 'may', 'mf', 'p']
```


```python
def clean_text(text, parts_of_speech=['ADJ' ,'NOUN', 'ADV', 'VERB', 'PROPN'],
              remove_sw=True, sw=stops):
    """
    Cleans text and filters according to part of speech.
    
    Parameters
    ----------
    text : str
    
    parts_of_speech : list of strings
        refer to parts of speech in SpaCy
        
    remove_sw : bool
    
    sw : list of strings
        add your own if necessary
        
    Returns
    -------
    out3 : str
        string with parts of speech filtered
    """
    # cleaning
    text = text.lower()
    text = text.replace('[removed]', '')
    text = text.replace('[deleted]', '')
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(r'[^\w\s]+', ' ', text)
    text = re.sub("p*\d", "", text)
    text = re.sub(r" +", ' ', text)
    
    # pass text into nlp then remove stopwords
    text = nlp(text)
    
    # .lemma_ and .pos_ are helpful extracting the lemmatized
    # word and part of speech.
    
    out = []
    for token in text:
         out.append((token.lemma_, token.pos_))
    poss = parts_of_speech
    
    out3 = ''
    for item in out:
        if item[1] in poss:
            out3 = out3 + ' ' + item[0]
    
    if remove_sw:
        dummy = out3.split()
        dummy = [word for word in dummy if word not in sw]
        out3 = ' '.join(dummy)
        return set(out3.strip().split())
    
    else:
        return set(out3.strip().split())
```


```python
def tabulate(corpus, parts_of_speech=['ADJ' ,'NOUN', 'ADV', 'VERB', 'PROPN'],
              remove_sw=True, sw=stops):
    """
    Converts phr4r raw text into a table.
    
    Parameters
    ----------
    Corpus : list of strings
    
    Returns
    -------
    df : pandas DataFrame
    """
    age = []
    sex = []
    so = []
    document = []

    for doc in corpus:
        try:
            age_ = re.findall(r"^(\d{2})", doc)[0]
            doc = doc.replace(age_, '')
        except:
            age_ = '0'
            
        try:
            stat = re.findall(r"(\[?[MmFf]4[MmFfAa]\]?)", doc)[0]
            doc = doc.replace(stat, '')
            stat = re.findall(r"\[?([MmFf])4([MmFfAa])\]?", stat)[0]
            stat = (stat[0].upper(), stat[1].upper())
        except:
            stat = ('Unknown', 'Unknown')
        
        age.append(int(age_))
        sex.append(stat[0])

        if stat[0]=='Unknown' or stat[1]=='Unknown':
            so.append("Unknown")
        elif stat[0].lower()==stat[1].lower():
            so.append("Homosexual")
        elif stat[1].lower()=="a":
            so.append("Indeterminate")
        else:
            so.append("Straight")
        
        word_set = clean_text(doc, parts_of_speech=parts_of_speech)
        document.append(word_set)
    
    df = pd.DataFrame({"Age": age, "Sex": sex, "Sexual Orientation": so, 
                       "Word Set": document})
    return df
```

#### Combine all reddit texts into a list of texts


```python
all_txt = df_all['texts'].values
```


```python
len(all_txt)
```




    61679



#### Preprocess data and extract only `verbs`, `adjectives` and `nouns` (including `proper nouns`)


```python
df = tabulate(all_txt, parts_of_speech=['ADJ' ,'NOUN', 'VERB', 'PROPN'])
```


```python
df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{bi, old, place, fun, manila, sta, mesa, age}</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>M</td>
      <td>Straight</td>
      <td>{drink, hohol, nsa}</td>
    </tr>
    <tr>
      <td>2</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{movie, chill, shaw}</td>
    </tr>
    <tr>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{buddy, new, time, take, try, manila, come, re...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{san, hangout, pablo, laguna}</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>95</td>
      <td>27</td>
      <td>M</td>
      <td>Straight</td>
      <td>{hohol, like, cocol}</td>
    </tr>
    <tr>
      <td>96</td>
      <td>24</td>
      <td>M</td>
      <td>Straight</td>
      <td>{cunnilingu, lean, manila, work, lay, girl, se...</td>
    </tr>
    <tr>
      <td>97</td>
      <td>0</td>
      <td>M</td>
      <td>Straight</td>
      <td>{need, area, fill, taft}</td>
    </tr>
    <tr>
      <td>98</td>
      <td>0</td>
      <td>F</td>
      <td>Indeterminate</td>
      <td>{bvs, manila, inbox, watch, theater, optional,...</td>
    </tr>
    <tr>
      <td>99</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{massage, good, herb, sex, stupid, hour, bonus...</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>




```python
df.shape
```




    (61679, 4)



#### Investigate quality of data


```python
df['Sex'].unique()
```




    array(['M', 'F', 'Unknown'], dtype=object)




```python
df['Sexual Orientation'].unique()
```




    array(['Homosexual', 'Straight', 'Indeterminate', 'Unknown'], dtype=object)



**`Sex` and `Sexual Orientation` have the correct values.**


```python
sorted(df['Age'].unique().tolist())[:10]

# false alarms: 1, 10, 12 -> change to 0

# real: 11 -> no change
```




    [0, 1, 10, 11, 12, 15, 16, 17, 18, 19]



**Ages `1, 10, 12` are not real ages (upon checking the original data), hence will be changed to 0. Unfortunately, `11` was stated as a real age.**


```python
df.loc[df.Age==1, 'Age'] = 0
df.loc[df.Age==10, 'Age'] = 0
df.loc[df.Age==12, 'Age'] = 0
```


```python
sorted(df['Age'].unique().tolist())[30:]
```




    [43, 44, 45, 46, 47, 48, 50, 51, 52, 59, 60, 62, 66, 69, 70, 88, 99]



**Age `66` was written as 666 by author, hence will be changed to 0**


```python
df.loc[[21554, 34276], 'Age'] = 0
```

#### Load processed data to pickle

```python
with open('df_adj_n_v_propn.pkl', 'wb') as f:
    pickle.dump(df, f)
```

#### Read original dataframes which include more columns


```python
with open('df_bz.pkl', 'rb') as f:
    df_bz = pickle.load(f)
    
with open('df_xz.pkl', 'rb') as f:
    df_xz = pickle.load(f)
    
with open('df_zst.pkl', 'rb') as f:
    df_zst = pickle.load(f)
    
df_all_cols = df_bz.append(df_xz).append(df_zst).reset_index(drop=True)
```

#### Convert unix timestamp to datetime


```python
df_all_cols['date'] = pd.to_datetime(df_all_cols['created_utc'], unit='s')
```

#### Join previous `df` with `df_all_cols`


```python
df_all_cols = df.join(df_all_cols)
```


```python
df_all_cols.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{bi, old, place, fun, manila, sta, mesa, age}</td>
      <td>phr4r</td>
      <td>1451891616</td>
      <td>EdWao</td>
      <td>[m4m] looking for FUN</td>
      <td>Hey. I'm 18 bi m from Sta. Mesa Manila. Lookin...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>8</td>
      <td>[m4m] looking for FUN\nHey. I'm 18 bi m from S...</td>
      <td>2016-01-04 07:13:36</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>M</td>
      <td>Straight</td>
      <td>{drink, hohol, nsa}</td>
      <td>phr4r</td>
      <td>1451950820</td>
      <td>[deleted]</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]</td>
      <td>[removed]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]...</td>
      <td>2016-01-04 23:40:20</td>
    </tr>
  </tbody>
</table>
</div>



#### Load dataframe with complete columns to pickle

```python
with open('df_all_cols.pkl', 'wb') as f:
    pickle.dump(df_all_cols, f)
```

## 3. Exploratory Data Analysis

Now that we have cleaned the data, we'll perform Exploratory Data Analysis to extract initial insights on the phr4r subreddit submissions.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import collections
```


```python
with open('df_adj_n_v_propn.pkl', 'rb') as pkl1:
    df1  = pickle.load(pkl1)
    
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{bi, age, sta, mesa, place, fun, manila, old}</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>M</td>
      <td>Straight</td>
      <td>{hohol, drink, nsa}</td>
    </tr>
    <tr>
      <td>2</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{shaw, movie, chill}</td>
    </tr>
    <tr>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{come, new, reply, time, girlfriend, inbox, ta...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{hangout, laguna, pablo, san}</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.shape
```




    (61679, 4)



#### Analysis on posts per sexual orientation


```python
so  = list(df1['Sexual Orientation'].unique())
```


```python
df1['Sexual Orientation'].value_counts(normalize=True)*100
```




    Straight         69.809822
    Indeterminate    17.592698
    Homosexual        6.358728
    Unknown           6.238752
    Name: Sexual Orientation, dtype: float64




```python
perc_so = df1['Sexual Orientation'].value_counts(normalize=True)*100

so = {}
so['Straight'] = perc_so[0]
so['Indeterminate'] = perc_so[1]
so['Homosexual'] = perc_so[2]
so['Unknown'] = perc_so[3]
```


```python
labels_so = []
vals_so = []
for i,j in so.items():
    labels_so.append(f'{i} {j: .4}%')
    vals_so.append(j)
```


```python
plt.rcParams['figure.facecolor'] = '#17202a'
# Create a circle for the center orf the plot
plt.figure(figsize=(8,7), dpi=150)
plt.rcParams['figure.facecolor'] = '#17202a'

circle=plt.Circle( (0,0), 0.5, color='#17202a')

# Give color names
plt.pie(vals_so, labels=labels_so, 
        colors=['#ff5700','#cee3f8','#5f99cf','#336699'],
       textprops={'color':"w",
                 'size':15})

p=plt.gcf()
p.gca().add_artist(circle)
plt.show()
```


![png](output_97_0.png)


Plot above shows the proportion of posts based on sexual orientation. **69.81%** of the posts were by users looking for the opposite sex.

#### Analysis on posts per gender


```python
df1['Sex'].value_counts(normalize=True)*100
```




    M          69.934662
    F          23.826586
    Unknown     6.238752
    Name: Sex, dtype: float64




```python
sx_so = df1['Sex'].value_counts(normalize=True)*100

sx = {}
sx['Male'] = sx_so[0]
sx['Female'] = sx_so[1]
sx['Unknown'] = sx_so[2]
```


```python
labels_sx = []
vals_sx = []
for i,j in sx.items():
    labels_sx.append(f'{i} {j: .4}%')
    vals_sx.append(j)
```


```python
plt.rcParams['figure.facecolor'] = '#17202a'

# Create a circle for the center orf the plot
plt.figure(figsize=(8,7), dpi=150)
plt.rcParams['figure.facecolor'] = '#17202a'

circle=plt.Circle( (0,0), 0.5, color='#17202a')

# Give color names
plt.pie(vals_sx, labels=labels_sx, 
        colors=['#336699','#ff5700','#cee3f8'],
       textprops={'color':"w",
                 'size':15})

p=plt.gcf()
p.gca().add_artist(circle)
plt.show()
```


![png](output_103_0.png)


Plot above shows the proportion of posts based on gender. **69.93%** of the submissions were posted by males.

#### Age distribution


```python
age_dist = list(df1[df1['Age'] != 0]['Age'])
```


```python
c = collections.Counter(age_dist)
c = sorted(c.items())
age_ = [i[0] for i in c]
age_freq =[i[1] for i in c]
```


```python
f, ax = plt.subplots(figsize=(15,5), dpi=150)
ax.set_facecolor('#17202a')

index = np.arange(len(age_))

plt.bar(index, age_freq, color='#ff5700')
plt.xlabel('Age', fontsize=10)
plt.ylabel('No of Posts', fontsize=10)
plt.xticks(index, age_, fontsize=10, rotation=45)
title = plt.title('Age distribution of r/phr4r post authors', fontsize=15)
plt.setp(title, color='white')


ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')


plt.show()
```


![png](output_108_0.png)


Get the IQR and subsequently the outliers


```python
q75, q25 = np.percentile(age_dist, [75 ,25])

iqr = q75 - q25

ub = q75 + 1.5*iqr #upperbound
lb = q25 - 1.5*iqr #lowerbound

print(f'upperbound {ub} y/o')
print(f'lowerbound {lb} y/o')
```

    upperbound 34.5 y/o
    lowerbound 14.5 y/o
    


```python
outliers=[]
for i in age_:
    if i < lb or i > ub:
        outliers.append(i)
```


```python
c_o = collections.Counter(outliers)
c_o = sorted(c_o.items())
age_o = [i[0] for i in c_o]
age_freq_o =[i[1] for i in c_o]
```


```python
f, ax = plt.subplots(figsize=(15,5), dpi=150)
ax.set_facecolor('#17202a')

index = np.arange(len(age_))

plt.bar(index, age_freq, color='#cee3f8')
plt.bar(9, 6767, color='#ff5700')


plt.xlabel('Age', fontsize=10)
plt.ylabel('No of Posts', fontsize=10)
plt.xticks(index, age_, fontsize=10, rotation=45)
title = plt.title('Age distribution of r/phr4r post authors', fontsize=15)
plt.setp(title, color='white')


ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')


plt.show()
```


![png](output_113_0.png)


Given the age distribution of the users in the r/phr4r subreddit seen above, it is apparent that the distribution is **left-skewed** with most of the users being of **23-year-old** users. 

The left-skewed distribution illustrates that the users are **15 to 35 years old**, which is also supported by the age distribution’s **interquartile range**.


```python
df_23 = df1[df1.Age == 23]
df_23.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{come, new, reply, time, girlfriend, inbox, ta...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>23</td>
      <td>M</td>
      <td>Indeterminate</td>
      <td>{area, españa}</td>
    </tr>
    <tr>
      <td>15</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{approach, lookig, fwb, fetish, woman, age}</td>
    </tr>
    <tr>
      <td>16</td>
      <td>23</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{katipunan, marikina, mind, buddy, pasig, area...</td>
    </tr>
    <tr>
      <td>19</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{reddit, r, killer, lurker, tress, discover, t...</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_23 = df_23['Sex'].value_counts(normalize=True)*100
age_23
```




    M          68.834048
    F          27.441998
    Unknown     3.723954
    Name: Sex, dtype: float64




```python
gen=[]
prop_gen=[]
for i,j in age_23.items():
    gen.append(i)
    prop_gen.append(j)
```


```python
f, ax = plt.subplots(figsize=(7,5), dpi=150)
ax.set_facecolor('#17202a')

plt.bar(gen, prop_gen, color=['#336699','#ff5700','#cee3f8'])

plt.xlabel('Sex', fontsize=10)
plt.ylabel('Proportion of posts', fontsize=10)
title = plt.title('Posts by users of age 23', fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')


plt.show()
```


![png](output_118_0.png)


#### Age distribution between straight and homosexual


```python
df_nz = df1[df1['Age'] != 0]
df_str = df_nz[df_nz['Sexual Orientation'] == 'Straight']
df_homo = df_nz[df_nz['Sexual Orientation'] == 'Homosexual']
df_ind = df_nz[df_nz['Sexual Orientation'] == 'Indeterminate']
df_unkwn = df_nz[df_nz['Sexual Orientation'] == 'Unknown']
```


```python
f, ax = plt.subplots(figsize=(15,5), dpi=150)
ax.set_facecolor('#17202a')

#homo
age_dist = list(df_homo['Age'])
c = collections.Counter(age_dist)
c = sorted(c.items())
age_ = [i[0] for i in c]
age_freq =[i[1] for i in c]
index = np.arange(len(age_))
plt.bar(index, (age_freq/np.sum(age_freq))*100, color='#ff5700',
        label='LGBT') #homo


#straight
age_dist_s = list(df_str['Age'])
c_s = collections.Counter(age_dist_s)
c_s = sorted(c_s.items())
age_s = [i[0] for i in c_s]
age_freq_s =[i[1] for i in c_s]
index_s = np.arange(len(age_s))
plt.bar(index_s, (age_freq_s/np.sum(age_freq_s))*100, color='#fafafa',
        alpha=0.7, label='Straight') #straight


plt.xlabel('Age', fontsize=10)
plt.ylabel('Perc of Posts', fontsize=10)
plt.xticks(index, age_, fontsize=10, rotation=45)
title = plt.title('Age distribution of r/phr4r: straight vs LGBT authors'
                  , fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_xlim(0, len(age_))
ax.legend(borderpad=.5, labelspacing=.5, fontsize=15)

plt.show()
```


![png](output_121_0.png)


Plot above shows age distribution of each sexual orientation. Interestingly, **homosexuals tend to post at a younger age than straight redditors**.

The generally younger population of LGBTQI+ may be in part due to the **anonymity granted by the platform**: users may simply be **exploring bisexual curiosities** or other things that would otherwise stigmatize them if expressed through other media.

Moreover, the comparative age distribution between the straight and homosexual users is indicative of the **homosexuals being more active on the platform at a younger age** whereas **active users that identify to be straight users tend to be active on the platform at a much latter age**. This observation is supported by a study by Barcz, M, et al., in 2019 which emphasized that **online anonymity** similar to what Reddit features **enhances self-expression and cultivates confidence in self-identification** especially in contrast to hostile environments similar to the prejudices and social constructs that the LGBTQI+ community faces in the Philippines. Given this Reddit platform, the youth of the LGBTQI+ community are provided with an avenue for them to openly interact and seek partners and peers without the worry of social dynamics and unjust playing into place.

#### Age distribution between genders


```python
df_nz = df1[df1['Age'] != 0]

df_f = df_nz[df_nz.Sex == 'F']
df_m = df_nz[df_nz.Sex == 'M']
df_u = df_nz[df_nz.Sex == 'Unknown']
```


```python
f, ax = plt.subplots(figsize=(15,5), dpi=150)
ax.set_facecolor('#17202a')

#female
age_dist_f = list(df_f['Age'])
c_f = collections.Counter(age_dist_f)
c_f = sorted(c_f.items())
age_f = [i[0] for i in c_f]
age_freq_f =[i[1] for i in c_f]
index = np.arange(len(age_f))
plt.plot(index, (age_freq_f/np.sum(age_freq_f))*100, color='#ff5700',
        label='Female', linewidth=3)

#male
age_dist_m = list(df_m['Age'])
c_m = collections.Counter(age_dist_m)
c_m = sorted(c_m.items())
age_m = [i[0] for i in c_m]
age_mreq_m =[i[1] for i in c_m]
index = np.arange(len(age_m))
plt.plot(index, (age_mreq_m/np.sum(age_mreq_m))*100, color='#5f99cf',
        label='Male', linewidth=3)

#unknown
age_dist_u = list(df_u['Age'])
c_u = collections.Counter(age_dist_u)
c_u = sorted(c_u.items())
age_u = [i[0] for i in c_u]
age_ureq_u =[i[1] for i in c_u]
index = np.arange(len(age_u))
plt.plot(index, (age_ureq_u/np.sum(age_ureq_u))*100, color='#fafafa',
        label='Unknown', linewidth=3)


plt.xlabel('Age', fontsize=10)
plt.ylabel('Perc of Posts', fontsize=10)
plt.xticks(index, age_, fontsize=10, rotation=45)
title = plt.title('Age distribution of r/phr4r by sex'
                  , fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_xlim(0, len(age_))
ax.legend(borderpad=.5, labelspacing=.5, fontsize=15)

plt.show()
```


![png](output_125_0.png)


Plot above shows age distribution by gender. We can see that **female authors tend to be younger than males**.

#### Distribution by gender and sexual orientation on different time intervals


```python
with open('df_all_cols.pkl', 'rb') as pkl:
    df_date  = pickle.load(pkl)
```


```python
df_date.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{bi, age, sta, mesa, place, fun, manila, old}</td>
      <td>phr4r</td>
      <td>1451891616</td>
      <td>EdWao</td>
      <td>[m4m] looking for FUN</td>
      <td>Hey. I'm 18 bi m from Sta. Mesa Manila. Lookin...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>8</td>
      <td>[m4m] looking for FUN\nHey. I'm 18 bi m from S...</td>
      <td>2016-01-04 07:13:36</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>M</td>
      <td>Straight</td>
      <td>{hohol, drink, nsa}</td>
      <td>phr4r</td>
      <td>1451950820</td>
      <td>[deleted]</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]</td>
      <td>[removed]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]...</td>
      <td>2016-01-04 23:40:20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{shaw, movie, chill}</td>
      <td>phr4r</td>
      <td>1452053372</td>
      <td>[deleted]</td>
      <td>30 [M4F] Shaw, Movie and chill</td>
      <td>[deleted]</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>30 [M4F] Shaw, Movie and chill\n[deleted]</td>
      <td>2016-01-06 04:09:32</td>
    </tr>
    <tr>
      <td>3</td>
      <td>23</td>
      <td>M</td>
      <td>Straight</td>
      <td>{come, new, reply, time, girlfriend, inbox, ta...</td>
      <td>phr4r</td>
      <td>1452065137</td>
      <td>ThreesomePH</td>
      <td>23 [M4F] Manila area. Actually looking for a t...</td>
      <td>So my girlfriend and I want to try something n...</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>8</td>
      <td>23 [M4F] Manila area. Actually looking for a t...</td>
      <td>2016-01-06 07:25:37</td>
    </tr>
    <tr>
      <td>4</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{hangout, laguna, pablo, san}</td>
      <td>phr4r</td>
      <td>1452248658</td>
      <td>[deleted]</td>
      <td>30 [M4F] San Pablo Laguna, hangout?</td>
      <td>[deleted]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>30 [M4F] San Pablo Laguna, hangout?\n[deleted]</td>
      <td>2016-01-08 10:24:18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_date_ = df_date[['Sex','Sexual Orientation','date']]
df_date_['wk_day'] = df_date_['date'].dt.dayofweek
df_date_['hr'] = df_date_['date'].dt.hour
df_date_.head()
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>date</th>
      <th>wk_day</th>
      <th>hr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>2016-01-04 07:13:36</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M</td>
      <td>Straight</td>
      <td>2016-01-04 23:40:20</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M</td>
      <td>Straight</td>
      <td>2016-01-06 04:09:32</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M</td>
      <td>Straight</td>
      <td>2016-01-06 07:25:37</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M</td>
      <td>Straight</td>
      <td>2016-01-08 10:24:18</td>
      <td>4</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
d_s = df_date_[df_date_['Sexual Orientation'] == 'Homosexual']
d_h = df_date_[df_date_['Sexual Orientation'] == 'Straight']

d_f = df_date_[df_date_.Sex == 'F']
d_m = df_date_[df_date_.Sex == 'M']
d_u = df_date_[df_date_.Sex == 'Unknown']
```


```python
f, ax = plt.subplots(figsize=(8,5), dpi=150)
ax.set_facecolor('#17202a')

#homo
h_hr_dist = list(d_h['hr'])
c = collections.Counter(h_hr_dist)
c = sorted(c.items())
h_hr_ = [i[0] for i in c]
h_hr_freq =[i[1] for i in c]
index = np.arange(len(h_hr_))
plt.plot(index, (h_hr_freq/np.sum(h_hr_freq))*100, color='#ff5700',
        label='LGBT',linewidth=2)


#straight
s_hr_dist = list(d_s['hr'])
c_ = collections.Counter(s_hr_dist)
c_ = sorted(c_.items())
s_hr_ = [i[0] for i in c_]
s_hr_freq =[i[1] for i in c_]
index = np.arange(len(s_hr_))
plt.plot(index, (s_hr_freq/np.sum(s_hr_freq))*100, color='#fafafa',
        label='Straights',linewidth=2)


plt.xlabel('Hour of day', fontsize=10)
plt.ylabel('Perc of Posts', fontsize=10)
plt.xticks(index, h_hr_, fontsize=10, rotation=45)
title = plt.title('Hourly distribution of r/phr4r: straight vs LGBT authors'
                  , fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_xlim(0, len(h_hr_))
ax.legend(borderpad=.5, labelspacing=.5, fontsize=10)

plt.show()
```


![png](output_132_0.png)


We can see a huge spike at **2-3PM** on number of posts for both straight and homosexuals.


```python
f, ax = plt.subplots(figsize=(10,5), dpi=150)
ax.set_facecolor('#17202a')

#male
age_dist_m = list(d_m['hr'])
c_m = collections.Counter(age_dist_m)
c_m = sorted(c_m.items())
age_m = [i[0] for i in c_m]
age_mreq_m =[i[1] for i in c_m]
index = np.arange(len(age_m))
plt.plot(index, (age_mreq_m/np.sum(age_mreq_m))*100, color='#5f99cf',
        label='Male', linewidth=3)

#female
age_dist_f = list(d_f['hr'])
c_f = collections.Counter(age_dist_f)
c_f = sorted(c_f.items())
age_f = [i[0] for i in c_f]
age_freq_f =[i[1] for i in c_f]
index = np.arange(len(age_f))
plt.plot(index, (age_freq_f/np.sum(age_freq_f))*100, color='#ff5700',
        label='Female', linewidth=3)


plt.xlabel('Hour of day', fontsize=10)
plt.ylabel('Perc of Posts', fontsize=10)
plt.xticks(index, h_hr_, fontsize=10, rotation=45)
title = plt.title('Hourly distribution of r/phr4r: female vs male authors'
                  , fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_xlim(0, len(h_hr_))
ax.legend(borderpad=.5, labelspacing=.5, fontsize=10)

plt.show()
```


![png](output_134_0.png)


Same results for both genders: **2-3PM**


```python
f, ax = plt.subplots(figsize=(8,5), dpi=150)
ax.set_facecolor('#17202a')

#homo
h_wk_dist = list(d_h['wk_day'])
c = collections.Counter(h_wk_dist)
c = sorted(c.items())
h_wk_ = [i[0] for i in c]
h_wk_freq =[i[1] for i in c]
index = np.arange(len(h_wk_))
plt.plot(index, (h_wk_freq/np.sum(h_wk_freq))*100, color='#ff5700',
        label='LGBT',linewidth=3)


#straight
s_wk_dist = list(d_s['wk_day'])
c_ = collections.Counter(s_wk_dist)
c_ = sorted(c_.items())
s_wk_ = [i[0] for i in c_]
s_wk_freq =[i[1] for i in c_]
index = np.arange(len(s_wk_))
plt.plot(index, (s_wk_freq/np.sum(s_wk_freq))*100, color='#fafafa',
        label='Straights', linewidth=3)

wk_day_list = ['Monday','Tuesday','Wednesday',
               'Thursday','Friday','Saturday','Sunday']

plt.xlabel('Day of week', fontsize=10)
plt.ylabel('Perc of Posts', fontsize=10)
plt.xticks(index, wk_day_list, fontsize=10)
title = plt.title('Daily distribution of r/phr4r: straight vs LGBT authors'
                  , fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_xlim(0, len(h_wk_)-1)
ax.legend(borderpad=.5, labelspacing=.5, fontsize=10)

plt.show()
```


![png](output_136_0.png)


For **homosexuals**, there is a **steady number of posts during the weekday** then starts to **go up on the end of the week, peaking on Saturday**, and finally **goes down again on Sunday**.

However, for **straights**, the number of posts **fluctuates**, but has the **highest number during Wednesdays**.


```python
f, ax = plt.subplots(figsize=(8,5), dpi=150)
ax.set_facecolor('#17202a')

#male
age_dist_m = list(d_m['wk_day'])
c_m = collections.Counter(age_dist_m)
c_m = sorted(c_m.items())
age_m = [i[0] for i in c_m]
age_mreq_m =[i[1] for i in c_m]
index = np.arange(len(age_m))
plt.plot(index, (age_mreq_m/np.sum(age_mreq_m))*100, color='#5f99cf',
        label='Male', linewidth=3)

#female
age_dist_f = list(d_f['wk_day'])
c_f = collections.Counter(age_dist_f)
c_f = sorted(c_f.items())
age_f = [i[0] for i in c_f]
age_freq_f =[i[1] for i in c_f]
index = np.arange(len(age_f))
plt.plot(index, (age_freq_f/np.sum(age_freq_f))*100, color='#ff5700',
        label='Female', linewidth=3)

wk_day_list = ['Monday','Tuesday','Wednesday',
               'Thursday','Friday','Saturday','Sunday']

plt.xlabel('Day of week', fontsize=10)
plt.ylabel('Perc of Posts', fontsize=10)
plt.xticks(index, wk_day_list, fontsize=10, rotation=45)
title = plt.title('Daily distribution of r/phr4r: straight vs homosexual authors'
                  , fontsize=15)
plt.setp(title, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_xlim(0, len(h_wk_)-1)
ax.legend(borderpad=.5, labelspacing=.5, fontsize=10)

plt.show()
```


![png](output_138_0.png)


For **males**, there is a **steady number of posts during the weekday** then starts to **go up on the end of the week, peaking on Friday/Saturday**, and finally **goes down again on Sunday**.

However, for **females**, the number of posts **fluctuates**, but **peaks highest during Saturdays**.

## 4. Profiling

For this study, we aim to compare the behavior of each sexual orientation, straight and homosexuals, by performing analysis on each separately to discover underlying insights that may differ or be similar to both classes.


```python
import pickle
import pandas as pd
```

#### Load data to dataframe


```python
with open('df_all_cols.pkl', 'rb') as f:
    df_all_cols = pickle.load(f)
```

#### Create dataframes for each sexual orientation


```python
df_all_cols['Sexual Orientation'].unique()
```




    array(['Homosexual', 'Straight', 'Indeterminate', 'Unknown'], dtype=object)




```python
df_all_cols_hom = df_all_cols[df_all_cols['Sexual Orientation']=='Homosexual']
df_all_cols_str = df_all_cols[df_all_cols['Sexual Orientation']=='Straight']
```


```python
df_all_cols_hom.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{old, mesa, sta, manila, bi, age, fun, place}</td>
      <td>phr4r</td>
      <td>1451891616</td>
      <td>EdWao</td>
      <td>[m4m] looking for FUN</td>
      <td>Hey. I'm 18 bi m from Sta. Mesa Manila. Lookin...</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>8</td>
      <td>[m4m] looking for FUN\nHey. I'm 18 bi m from S...</td>
      <td>2016-01-04 07:13:36</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0</td>
      <td>M</td>
      <td>Homosexual</td>
      <td>{experiment, complicated, lad, manila, make, n...</td>
      <td>phr4r</td>
      <td>1452748498</td>
      <td>toohot888</td>
      <td>[m4m] Bi-curios lad here. Near Mendiola</td>
      <td>I have a complicated situation. I made a new r...</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>3</td>
      <td>[m4m] Bi-curios lad here. Near Mendiola\nI hav...</td>
      <td>2016-01-14 05:14:58</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_all_cols_str.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Sexual Orientation</th>
      <th>Word Set</th>
      <th>subreddit</th>
      <th>created_utc</th>
      <th>author</th>
      <th>title</th>
      <th>selftext</th>
      <th>ups</th>
      <th>downs</th>
      <th>score</th>
      <th>num_comments</th>
      <th>texts</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>M</td>
      <td>Straight</td>
      <td>{hohol, nsa, drink}</td>
      <td>phr4r</td>
      <td>1451950820</td>
      <td>[deleted]</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]</td>
      <td>[removed]</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>[M4F] Let's have a drink? [HOHOL/NSA/Anything]...</td>
      <td>2016-01-04 23:40:20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>30</td>
      <td>M</td>
      <td>Straight</td>
      <td>{movie, chill, shaw}</td>
      <td>phr4r</td>
      <td>1452053372</td>
      <td>[deleted]</td>
      <td>30 [M4F] Shaw, Movie and chill</td>
      <td>[deleted]</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>30 [M4F] Shaw, Movie and chill\n[deleted]</td>
      <td>2016-01-06 04:09:32</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Data Mining

To further discover hidden insights on the posts and demographics of phr4r submissions, we will use Frequent Itemset Mining (FIM) and Association Rule Mining (ARM) wherein each post is treated as a transaction, and the words extracted from the post (verbs, adjectives, and nouns) are their items.

All methods below were done using Spark.

### a. Frequent Itemset Mining

Frequent Itemset Mining is a subset of data mining that attempts to discover interesting and useful patterns in a transaction database. (Fournier‐Viger, 2016) Groups of items that appear together frequently (itemsets) based on a support level is generated. Support is computed as the total count of how many of the itemsets appear in the transactions (N). Relative support is expressed as a percent of the total number of transactions in the database (N) that cover a candidate itemset (X). Confidence is the likeliness of occurrence of consequent on the cart given that the cart already has the antecedents. This could be computed by the support of candidate itemsets (X and Y) divided by the support of candidate (X).  

#### <center>𝑋 𝑠𝑢𝑝 (𝑋 ∪ 𝑌) 𝑠𝑢𝑝(𝑋) = 𝑁, 𝑐𝑜𝑛𝑓(𝑋 → 𝑌) = 𝑠𝑢𝑝(𝑋)</center>

The package used for the FIM analysis is PyFIM. PyFIM is an extension module that makes several frequent itemset mining implementations such as Apriori, Eclat, FP-growth, etc., available as functions in Python. Finally, there is a function arules which can be used to generate the association rules. (http://www.borgelt.net/pyfim.html)  

For the analysis, FP-growth was chosen as the algorithm. FP-growth is a pattern-growth algorithm that scans a database for itemsets (with an identified minimum support) and creating a projected database from the itemsets that satisfy the aforementioned condition (Fournier‐Viger, 2016). This approach is based on divide and conquer strategy for producing the frequent item sets. A summary of the algorithm as pseudo code is shown below:
<img src="https://github.com/jeddahlyngacera/reddit-phr4r/blob/master/img1.PNG">
<center>Figure 3 – FP-growth Algorithm Pseudo-code (Fournier-Viger, 2018)</center>

The following parameters were used in the implementation of FP-growth (fim.fpgrowth):  

`Straight:`
```python
FPGrowth(itemsCol='Word Set', minSupport=0.001, minConfidence=0)
```
<pre>𝑠𝑢𝑝 = 0.1, 𝑐𝑜𝑛𝑓 = 0</pre>


`Homosexual:`
```python
FPGrowth(itemsCol='Word Set', minSupport=0.002, minConfidence=0)
```
<pre>𝑠𝑢𝑝 = 0.2, 𝑐𝑜𝑛𝑓 = 0</pre>


𝑤h𝑒𝑟𝑒 𝑠𝑢𝑝 𝑖𝑠 𝑡h𝑒 𝑚𝑖𝑛𝑖𝑚𝑢𝑚 𝑟𝑒𝑙𝑎𝑡𝑖𝑣𝑒 𝑠𝑢𝑝𝑝𝑜𝑟𝑡, 𝑎𝑛𝑑 𝑐𝑜𝑛𝑓 𝑖𝑠 𝑡h𝑒 𝑚𝑖𝑛𝑖𝑚𝑢𝑚 𝑐𝑜𝑛𝑓𝑖𝑑𝑒𝑛𝑐𝑒 𝑜𝑓 𝑎𝑛 𝑎𝑠𝑠𝑜𝑐𝑖𝑎𝑡𝑖𝑜𝑛 𝑟𝑢𝑙𝑒


```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import udf, desc
from pyspark.ml.fpm import FPGrowth
```

#### FIM for Straight orientation


```python
df_s = df_all_cols_str[['Word Set']].reset_index()

df_s['Word Set'] = df_s['Word Set'].apply(lambda x: list(x))
df_s['no_items'] = df_s['Word Set'].apply(lambda x: len(x))

df_s = df_s[df_s['no_items'] > 0].drop(columns=['no_items'])
```


```python
df_s.shape
```




    (42226, 2)




```python
df_s.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Word Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>[hohol, nsa, drink]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>[movie, chill, shaw]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>[girlfriend, buddy, try, manila, inbox, come, ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>[san, pablo, laguna, hangout]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>6</td>
      <td>[redditor, nga, halong, sini, kag, iloilo, din...</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfs = spark.createDataFrame(df_s[['index', 'Word Set']])
```


```python
dfs.show(5)
```

    +-----+--------------------+
    |index|            Word Set|
    +-----+--------------------+
    |    1| [hohol, nsa, drink]|
    |    2|[movie, chill, shaw]|
    |    3|[girlfriend, budd...|
    |    4|[san, pablo, lagu...|
    |    6|[redditor, nga, h...|
    +-----+--------------------+
    only showing top 5 rows
    
    


```python
fpgrowth1 = FPGrowth(itemsCol='Word Set', minSupport=0.001, minConfidence=0)
fpgrowth_trained1 = fpgrowth1.fit(dfs)
```


```python
freq_items_s = fpgrowth_trained1.freqItemsets
assoc_rules_s = fpgrowth_trained1.associationRules
conseqs_s = fpgrowth_trained1.transform(dfs)
```


```python
no_items = udf(lambda x: len(x))
freq_itemsS = freq_items_s.withColumn('no_items', 
                                      no_items(freq_items_s['items']))
```


```python
df_freq_itemsS = freq_itemsS.toPandas()
df_freq_itemsS['no_items'] = df_freq_itemsS['no_items'].astype('int')
```


```python
df_freq_itemsS.shape
```




    (82379, 3)



```python
with open('df_freq_items_str.pkl', 'wb') as f:
    pickle.dump(df_freq_itemsS, f)
```

#### Investigate frequent n-itemsets (less n+1-itemsets)


```python
itemset1 = df_freq_itemsS[df_freq_itemsS['no_items']==1
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset2 = df_freq_itemsS[df_freq_itemsS['no_items']==2
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset3 = df_freq_itemsS[df_freq_itemsS['no_items']==3
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset4 = df_freq_itemsS[df_freq_itemsS['no_items']==4
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset5 = df_freq_itemsS[df_freq_itemsS['no_items']==5
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset6 = df_freq_itemsS[df_freq_itemsS['no_items']==6
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
```


```python
words1 = []
for i in itemset1.loc[:9, 'items'].values:
    words1 += i

words1_ = set(words1)
print(sorted(words1_), '\n', len(words1_))
print(sorted(set(words1)), '\n', len(words1))
```

    ['area', 'buddy', 'chat', 'fun', 'makati', 'manila', 'pm', 'qc', 'talk', 'time'] 
     10
    ['area', 'buddy', 'chat', 'fun', 'makati', 'manila', 'pm', 'qc', 'talk', 'time'] 
     10
    


```python
words2 = []
for i in itemset2.loc[:12, 'items'].values:
    words2 += i

words2_ = set(words2) - words1_
print(sorted(words2_), '\n', len(words2_))
print(sorted(set(words2)), '\n', len(words2))
```

    ['fubu', 'fwb', 'good', 'like', 'metro', 'movie', 'need', 'thing', 'watch', 'work'] 
     10
    ['area', 'chat', 'fubu', 'fwb', 'good', 'like', 'manila', 'metro', 'movie', 'need', 'qc', 'talk', 'thing', 'time', 'watch', 'work'] 
     26
    


```python
words3 = []
for i in itemset3.loc[:45, 'items'].values:
    words3 += i

words3_ = set(words3) - set(words2)
print(sorted(words3_), '\n', len(words3_))
print(sorted(set(words3)), '\n', len(words3))
```

    ['friend', 'love', 'make', 'meet', 'new', 'old', 'people', 'post', 'would', 'year'] 
     10
    ['chat', 'friend', 'good', 'like', 'love', 'make', 'meet', 'movie', 'new', 'old', 'people', 'post', 'talk', 'thing', 'time', 'watch', 'work', 'would', 'year'] 
     138
    


```python
words4 = []
for i in itemset4.loc[:280, 'items'].values:
    words4 += i

words4_ = set(words4) - set(words3)
print(sorted(words4_), '\n', len(words4_))
print(sorted(set(words4)), '\n', len(words4))
```

    ['day', 'feel', 'life', 'lot', 'open', 'read', 'say', 'see', 'send', 'try'] 
     10
    ['chat', 'day', 'feel', 'friend', 'good', 'life', 'like', 'lot', 'love', 'make', 'meet', 'new', 'old', 'open', 'people', 'post', 'read', 'say', 'see', 'send', 'talk', 'thing', 'time', 'try', 'work', 'would', 'year'] 
     1124
    


```python
words5 = []
for i in itemset5.loc[:800, 'items'].values:
    words5 += i

words5_ = set(words5) - set(words4)
print(sorted(words5_), '\n', len(words5_))
print(sorted(set(words5)), '\n', len(words5))
```

    ['ask', 'bit', 'com', 'comment', 'game', 'https', 'phrr', 'reddit', 'tell', 'thank', 'www'] 
     11
    ['ask', 'bit', 'chat', 'com', 'comment', 'day', 'feel', 'free', 'friend', 'game', 'good', 'https', 'life', 'like', 'lot', 'love', 'make', 'meet', 'new', 'old', 'open', 'people', 'phrr', 'post', 'read', 'reddit', 'say', 'see', 'talk', 'tell', 'thank', 'thing', 'think', 'time', 'work', 'would', 'www', 'year'] 
     4005
    


```python
words6 = []
for i in itemset6.loc[:5000, 'items'].values:
    words6 += i

words6_ = set(words6) - set(words5)
print(sorted(words6_), '\n', len(words6_))
print(sorted(set(words6)), '\n', len(words6))
```

    [] 
     0
    ['feel', 'friend', 'good', 'meet', 'new', 'people', 'post', 'talk', 'thing', 'time', 'work'] 
     36
    

#### FIM for Homosexual orientation


```python
df_h = df_all_cols_hom[['Word Set']].reset_index()

df_h['Word Set'] = df_h['Word Set'].apply(lambda x: list(x))
df_h['no_items'] = df_h['Word Set'].apply(lambda x: len(x))

df_h = df_h[df_h['no_items'] > 0].drop(columns=['no_items'])
```


```python
df_h.shape
```




    (3824, 2)




```python
df_h.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Word Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>[old, mesa, sta, manila, bi, age, fun, place]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8</td>
      <td>[experiment, complicated, lad, manila, make, n...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11</td>
      <td>[sure, ganun, individual, fondle, trade, talks...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>16</td>
      <td>[buddy, hang, bore, antipolo, area, available,...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>57</td>
      <td>[people, feel, redditor, rr, kik, local, free,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfh = spark.createDataFrame(df_h[['index', 'Word Set']])
```


```python
dfh.show(5)
```

    +-----+--------------------+
    |index|            Word Set|
    +-----+--------------------+
    |    0|[old, mesa, sta, ...|
    |    8|[experiment, comp...|
    |   11|[sure, ganun, ind...|
    |   16|[buddy, hang, bor...|
    |   57|[people, feel, re...|
    +-----+--------------------+
    only showing top 5 rows
    
    


```python
fpgrowth2 = FPGrowth(itemsCol='Word Set', minSupport=0.002, minConfidence=0)
fpgrowth_trained2 = fpgrowth2.fit(dfh)
```


```python
freq_items_h = fpgrowth_trained2.freqItemsets
assoc_rules_h = fpgrowth_trained2.associationRules
conseqs_h = fpgrowth_trained2.transform(dfh)
```


```python
no_items = udf(lambda x: len(x))
freq_itemsH = freq_items_h.withColumn('no_items', 
                                      no_items(freq_items_h['items']))
```


```python
df_freq_itemsH = freq_itemsH.toPandas()
df_freq_itemsH['no_items'] = df_freq_itemsH['no_items'].astype('int')
```


```python
df_freq_itemsH.shape
```




    (14719, 3)



```python
with open('df_freq_items_hom.pkl', 'wb') as f:
    pickle.dump(df_freq_itemsH, f)
```

#### Investigate frequent n-itemsets (less n+1-itemsets)


```python
itemset1 = df_freq_itemsH[df_freq_itemsH['no_items']==1
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset2 = df_freq_itemsH[df_freq_itemsH['no_items']==2
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset3 = df_freq_itemsH[df_freq_itemsH['no_items']==3
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset4 = df_freq_itemsH[df_freq_itemsH['no_items']==4
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
itemset5 = df_freq_itemsH[df_freq_itemsH['no_items']==5
                         ].sort_values(['freq', 'no_items'], ascending=False
                                      ).reset_index(drop=True)
```


```python
words1 = []
for i in itemset1.loc[:9, 'items'].values:
    words1 += i

words1_ = set(words1)
print(sorted(words1_), '\n', len(words1_))
print(sorted(set(words1)), '\n', len(words1))
```

    ['buddy', 'friend', 'fun', 'girl', 'interested', 'makati', 'manila', 'pm', 'qc', 'talk'] 
     10
    ['buddy', 'friend', 'fun', 'girl', 'interested', 'makati', 'manila', 'pm', 'qc', 'talk'] 
     10
    


```python
words2 = []
for i in itemset2.loc[:12, 'items'].values:
    words2 += i

words2_ = set(words2) - words1_
print(sorted(words2_), '\n', len(words2_))
print(sorted(set(words2)), '\n', len(words2))
```

    ['bi', 'curious', 'first', 'like', 'message', 'send', 'telegram', 'time', 'try', 'would'] 
     10
    ['bi', 'curious', 'first', 'friend', 'girl', 'interested', 'like', 'message', 'pm', 'send', 'talk', 'telegram', 'time', 'try', 'would'] 
     26
    


```python
words3 = []
for i in itemset3.loc[:25, 'items'].values:
    words3 += i

words3_ = set(words3) - set(words2)
print(sorted(words3_), '\n', len(words3_))
print(sorted(set(words3)), '\n', len(words3))
```

    ['feel', 'good', 'make', 'old', 'post', 'see', 'thing', 'think', 'work', 'year'] 
     10
    ['feel', 'friend', 'girl', 'good', 'interested', 'like', 'make', 'message', 'old', 'post', 'see', 'send', 'talk', 'telegram', 'thing', 'think', 'time', 'try', 'work', 'would', 'year'] 
     78
    


```python
words4 = []
for i in itemset4.loc[:55, 'items'].values:
    words4 += i

words4_ = set(words4) - set(words3)
print(sorted(words4_), '\n', len(words4_))
print(sorted(set(words4)), '\n', len(words4))
```

    ['average', 'body', 'chat', 'need', 'people', 'place', 'pm', 'safe', 'sex', 'type'] 
     10
    ['average', 'body', 'chat', 'feel', 'friend', 'girl', 'good', 'interested', 'like', 'make', 'need', 'old', 'people', 'place', 'pm', 'post', 'safe', 'see', 'send', 'sex', 'talk', 'telegram', 'thing', 'think', 'time', 'try', 'type', 'work', 'would'] 
     224
    


```python
words5 = []
for i in itemset5.loc[:15, 'items'].values:
    words5 += i

words5_ = set(words5) - set(words4)
print(sorted(words5_), '\n', len(words5_))
print(sorted(set(words5)), '\n', len(words5))
```

    ['clean', 'decent', 'drink', 'fit', 'kami', 'message', 'new', 'open', 'willing', 'year'] 
     10
    ['average', 'body', 'clean', 'decent', 'drink', 'feel', 'fit', 'friend', 'good', 'kami', 'like', 'make', 'message', 'need', 'new', 'old', 'open', 'safe', 'see', 'send', 'sex', 'talk', 'thing', 'time', 'type', 'willing', 'work', 'year'] 
     80
    

### b. Association Rule Mining

In data mining, association rule learning is a widely accepted and well researched method for discovering relationships between variables in large databases. The main goal is to identify strong rules discovered in databases using different measures. A typical and widely-used example of association rule mining is Market Basket Analysis. The problem is to generate all association rules that have support and confidence greater than the user-specified minimum support and minimum confidence (Arora, et al 2013).

An association rule has two parts: an antecedent (IF) and a consequent (THEN). An antecedent is an item found within the data. A consequent is an item found in combination with the antecedent. 

**<center>IF Condition THEN Conclusion</center>**


```python
df_str = df_freq_itemsS.copy()
df_str['rel_sup'] = df_str.freq / df_str_.shape[0]

df_hom = df_freq_itemsH.copy()
df_hom['rel_sup'] = df_hom.freq / df_hom_.shape[0]
```

#### Get the top 50 most frequent itemsets for each orientation


```python
top50_1_str = [i[0] for i in df_str[df_str.no_items==1
                                   ].sort_values('rel_sup', ascending=False
                                                    )[:50]['items'].values]
top50_1_hom = [i[0] for i in df_hom[df_hom.no_items==1
                                   ].sort_values('rel_sup', ascending=False
                                                    )[:50]['items'].values]
```

#### Identify the difference and the intersection of the frequent itemsets


```python
str_ = sorted(set(top50_1_str) - set(top50_1_hom))
hom_ = sorted(set(top50_1_hom) - set(top50_1_str))
inter = sorted(set(top50_1_str).intersection(set(top50_1_hom)))
```


```python
pop = {}
for i, j in enumerate(top50_1_str):
    if j in top50_1_hom:
        k = top50_1_hom.index(j)
        pop[j] = i+k
```

#### Choose words that are frequent to both orientations


```python
sorted_x = sorted(pop.items(), key=lambda kv: kv[1])
```


```python
words = [i[0] for i in sorted_x]
```


```python
print(words)
```

    ['talk', 'qc', 'manila', 'buddy', 'area', 'fun', 'makati', 'pm', 'chat', 'time', 'friend', 'interested', 'need', 'good', 'girl', 'telegram', 'work', 'try', 'tele', 'send', 'message', 'would', 'like', 'date', 'hit', 'tonight', 'meet', 'make', 'sex', 'love', 'dm', 'momol', 'place', 'see', 'thing', 'new', 'watch', 'post']
    

#### Investigate a subset of those words


```python
words_sub = ['qc', 'manila', 'buddy', 'fun', 'makati', 'friend', 'girl', 
             'date', 'tonight', 'meet', 'sex', 'love', 'momol', 'place', 
             'watch']
```

#### Create dataframe where these words are either the antecedent or consequent


```python
df = pd.DataFrame(columns=['antecedent', 'consequent', 'confidence', 'lift', 'so', 'from', 'word'])
```


```python
for i in range(len(words_sub)):
    
    word = [words_sub[i]]
    
    ar_s_csq = assoc_rules_s.rdd.filter(lambda x: x['consequent']==word
                                       ).toDF().orderBy(desc('confidence')
                                                       ).limit(5).toPandas()
    ar_s_csq['so'] = 'Straight'
    ar_s_csq['from'] = 'Consequent'
    ar_s_csq['word'] = word[0]
    
    ar_s_a = assoc_rules_s.rdd.filter(lambda x: word[0] in x['antecedent']
                                     ).toDF().orderBy(desc('confidence')
                                                     ).limit(5).toPandas()
    ar_s_a['so'] = 'Straight'
    ar_s_a['from'] = 'Antecedent'
    ar_s_a['word'] = word[0]
    
    ar_h_csq = assoc_rules_h.rdd.filter(lambda x: x['consequent']==word
                                       ).toDF().orderBy(desc('confidence')
                                                       ).limit(5).toPandas()
    ar_h_csq['so'] = 'Homosexual'
    ar_h_csq['from'] = 'Consequent'
    ar_h_csq['word'] = word[0]
    
    ar_h_a = assoc_rules_h.rdd.filter(lambda x: word[0] in x['antecedent']
                                     ).toDF().orderBy(desc('confidence')
                                                     ).limit(5).toPandas()
    ar_h_a['so'] = 'Homosexual'
    ar_h_a['from'] = 'Antecedent'
    ar_h_a['word'] = word[0]
    
    df = df.append(ar_s_csq).append(ar_h_csq).append(ar_s_a).append(ar_h_a)
```


```python
df = df.reset_index(drop=True)
```

#### Check if all words in `words_sub` has been added to the dataframe


```python
for i in words_sub:
    print(i, df[df.word==i].shape)
```

    qc (20, 7)
    manila (20, 7)
    buddy (20, 7)
    fun (20, 7)
    makati (20, 7)
    friend (20, 7)
    girl (20, 7)
    date (20, 7)
    tonight (20, 7)
    meet (20, 7)
    sex (20, 7)
    love (20, 7)
    momol (20, 7)
    place (20, 7)
    watch (20, 7)
    

```python
with open('df_words_sub_ar.pkl', 'wb') as f:
    pickle.dump(df, f)
```

#### Sample results:
#### Example 1: word `fun` as `Consequent`


```python
df[(df.word=='fun') & (df['from']=='Consequent')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>60</td>
      <td>[interesting, cute]</td>
      <td>[fun]</td>
      <td>0.693548</td>
      <td>12.022075</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>61</td>
      <td>[people, thank, see]</td>
      <td>[fun]</td>
      <td>0.671875</td>
      <td>11.646385</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>62</td>
      <td>[much, game]</td>
      <td>[fun]</td>
      <td>0.657143</td>
      <td>11.391016</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>63</td>
      <td>[interesting, feel]</td>
      <td>[fun]</td>
      <td>0.652778</td>
      <td>11.315351</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>64</td>
      <td>[interesting, free]</td>
      <td>[fun]</td>
      <td>0.651515</td>
      <td>11.293464</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>65</td>
      <td>[average, open, discreet]</td>
      <td>[fun]</td>
      <td>1.000000</td>
      <td>10.863636</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>66</td>
      <td>[check, old]</td>
      <td>[fun]</td>
      <td>0.888889</td>
      <td>9.656566</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>67</td>
      <td>[fit, clean, send]</td>
      <td>[fun]</td>
      <td>0.833333</td>
      <td>9.053030</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>68</td>
      <td>[sex, see, good]</td>
      <td>[fun]</td>
      <td>0.818182</td>
      <td>8.888430</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>69</td>
      <td>[cute, clean]</td>
      <td>[fun]</td>
      <td>0.818182</td>
      <td>8.888430</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>fun</td>
    </tr>
  </tbody>
</table>
</div>



#### Example 2: word `fun` as `Antecedent`


```python
df[(df.word=='fun') & (df['from']=='Antecedent')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>70</td>
      <td>[people, friend, work, fun]</td>
      <td>[time]</td>
      <td>0.914894</td>
      <td>15.318120</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>71</td>
      <td>[string, fun]</td>
      <td>[attach]</td>
      <td>0.913793</td>
      <td>164.896699</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>72</td>
      <td>[live, year, fun]</td>
      <td>[work]</td>
      <td>0.897959</td>
      <td>16.176290</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>73</td>
      <td>[post, thing, love, fun]</td>
      <td>[time]</td>
      <td>0.895833</td>
      <td>14.998992</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>74</td>
      <td>[attach, fun]</td>
      <td>[string]</td>
      <td>0.883333</td>
      <td>178.467145</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>75</td>
      <td>[welcome, send, fun]</td>
      <td>[pic]</td>
      <td>1.000000</td>
      <td>30.349206</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>76</td>
      <td>[big, open, fun]</td>
      <td>[good]</td>
      <td>1.000000</td>
      <td>16.412017</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>77</td>
      <td>[type, clean, fun]</td>
      <td>[body]</td>
      <td>1.000000</td>
      <td>40.680851</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>78</td>
      <td>[type, good, fun]</td>
      <td>[body]</td>
      <td>1.000000</td>
      <td>40.680851</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
    <tr>
      <td>79</td>
      <td>[big, good, fun]</td>
      <td>[open]</td>
      <td>1.000000</td>
      <td>29.875000</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>fun</td>
    </tr>
  </tbody>
</table>
</div>



#### Example 3: word `sex` as both `Antecedent` and `Consequent`, filtered to only `Straight` orientation


```python
df[(df.word=='sex') & (df['so']=='Straight')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>200</td>
      <td>[phone, buddy]</td>
      <td>[sex]</td>
      <td>0.782609</td>
      <td>17.283700</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>201</td>
      <td>[phone]</td>
      <td>[sex]</td>
      <td>0.613483</td>
      <td>13.548608</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>202</td>
      <td>[host, love]</td>
      <td>[sex]</td>
      <td>0.367521</td>
      <td>8.116609</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>203</td>
      <td>[oral]</td>
      <td>[sex]</td>
      <td>0.350282</td>
      <td>7.735893</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>204</td>
      <td>[fuck, time]</td>
      <td>[sex]</td>
      <td>0.326241</td>
      <td>7.204947</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>210</td>
      <td>[metro, sex]</td>
      <td>[manila]</td>
      <td>0.721311</td>
      <td>9.681532</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>211</td>
      <td>[thing, sex, work]</td>
      <td>[time]</td>
      <td>0.681818</td>
      <td>11.415723</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>212</td>
      <td>[first, sex]</td>
      <td>[time]</td>
      <td>0.618557</td>
      <td>10.356533</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>213</td>
      <td>[sex, friend, time]</td>
      <td>[good]</td>
      <td>0.611111</td>
      <td>10.810548</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>214</td>
      <td>[thing, sex, good]</td>
      <td>[time]</td>
      <td>0.595238</td>
      <td>9.966108</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
  </tbody>
</table>
</div>



## IV. Results and Discussion

With the results from frequent itemset mining and association rule mining, let's explore and compare the most frequent and associated words per sexual orientation.

### 1. Frequent itemsets per sexual orientation


```python
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
%matplotlib inline
```

#### Load pickle files (cleaned data and frequent itemsets for each orientation)


```python
with open('df_adj_n_v_propn.pkl', 'rb') as f:
    df_ = pickle.load(f)
    
df_ = df_.reset_index()

df_['Word Set'] = df_['Word Set'].apply(lambda x: list(x))
df_['no_items'] = df_['Word Set'].apply(lambda x: len(x))

df_ = df_[df_['no_items'] > 0].drop(columns=['no_items'])
```


```python
df_str_ = df_[df_['Sexual Orientation']=='Straight']
df_hom_ = df_[df_['Sexual Orientation']=='Homosexual']
```


```python
with open('df_freq_items_str.pkl', 'rb') as f:
    df_str  = pickle.load(f)
    
df_str['rel_sup'] = df_str.freq / df_str_.shape[0]
top50_1_str = [i[0] for i in df_str[df_str.no_items==1].sort_values('rel_sup', ascending=False)[:50]['items'].values]
```


```python
with open('df_freq_items_hom.pkl', 'rb') as f:
    df_hom  = pickle.load(f)
    
df_hom['rel_sup'] = df_hom.freq / df_hom_.shape[0]
top50_1_hom = [i[0] for i in df_hom[df_hom.no_items==1].sort_values('rel_sup', ascending=False)[:50]['items'].values]
```

#### Determine frequent items/words that can be found on both, and those solely found in posts by each orientation


```python
str_ = sorted(set(top50_1_str) - set(top50_1_hom))
hom_ = sorted(set(top50_1_hom) - set(top50_1_str))
inter = sorted(set(top50_1_str).intersection(set(top50_1_hom)))

print('STRAIGHT:', str_, '\n')
print('HOMOSEXUAL:', hom_, '\n')
print('INTERSECTION:', inter, '\n')
```

    STRAIGHT: ['bgc', 'chill', 'coffee', 'cuddle', 'day', 'drink', 'fubu', 'fwb', 'hangout', 'movie', 'night', 'nsa'] 
    
    HOMOSEXUAL: ['bi', 'couple', 'curious', 'discreet', 'experience', 'feel', 'hang', 'hmu', 'open', 'pic', 'threesome', 'willing'] 
    
    INTERSECTION: ['area', 'buddy', 'chat', 'date', 'dm', 'friend', 'fun', 'girl', 'good', 'hit', 'interested', 'like', 'love', 'makati', 'make', 'manila', 'meet', 'message', 'momol', 'need', 'new', 'place', 'pm', 'post', 'qc', 'see', 'send', 'sex', 'talk', 'tele', 'telegram', 'thing', 'time', 'tonight', 'try', 'watch', 'work', 'would'] 
    
    


```python
wordcloud1 = WordCloud(width=1000, height=500, background_color='#17202a', 
                      stopwords=['Name', 'dtype', 'object'], colormap='Reds', 
                      min_font_size=11).generate(str(pd.DataFrame(str_, 
                                                columns=['words'])['words']))

wordcloud2 = WordCloud(width=1000, height=500, background_color='#17202a', 
                      stopwords=['Name', 'dtype', 'object', 'area'], colormap='Blues', 
                      min_font_size=11).generate(str(pd.DataFrame(inter, 
                                                columns=['words'])['words']))

wordcloud3 = WordCloud(width=1000, height=500, background_color='#17202a', 
                      stopwords=['Name', 'dtype', 'object'], colormap='gist_rainbow', 
                      min_font_size=11).generate(str(pd.DataFrame(hom_, 
                                                columns=['words'])['words']))
```


```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=1500, facecolor='#17202a')
ax1.imshow(wordcloud1, interpolation='bilinear')
ax1.axis('off')
ax2.imshow(wordcloud2, interpolation='bilinear')
ax2.axis('off')
ax3.imshow(wordcloud3, interpolation='bilinear')
ax3.axis('off')
fig.tight_layout(pad=-2)
fig.savefig('wc.png')
```


![png](output_228_0.png)


Wordcloud above shows frequent items/words used by each and both orientations in this order:

* **LEFT: straight**
* **MIDDLE: both**
* **RIGHT: homosexuals**

We can see that the posts made by each sexual orientation greatly differ in content. We can infer that **homosexuals tend to be more explicit** on what they are looking for in phr4r, unlike **straight redditors who still hold back** despite the guaranteed anonimity in reddit.

### 2. Association rules per sexual orientation


```python
import pandas as pd
import pickle
```

#### Load pickle file containing the association rules of words that are frequent to both orientations


```python
with open('df_words_sub_ar.pkl', 'rb') as f:
    df = pickle.load(f)
```

#### Filter to rules with `confidence of at least 90%`


```python
df_90 = df[df.confidence>=0.9]
```

#### Investigate words that may hold different rules per orientation

#### BUDDY:


```python
df_90[df_90.word=='buddy']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>40</td>
      <td>[read, open, like, chat]</td>
      <td>[buddy]</td>
      <td>0.934783</td>
      <td>10.539955</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>buddy</td>
    </tr>
    <tr>
      <td>45</td>
      <td>[fitness, gym]</td>
      <td>[buddy]</td>
      <td>1.000000</td>
      <td>11.623100</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>buddy</td>
    </tr>
    <tr>
      <td>50</td>
      <td>[lot, people, friend, buddy]</td>
      <td>[like]</td>
      <td>1.000000</td>
      <td>25.514199</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>buddy</td>
    </tr>
    <tr>
      <td>51</td>
      <td>[people, love, like, send, buddy]</td>
      <td>[friend]</td>
      <td>1.000000</td>
      <td>18.216566</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>buddy</td>
    </tr>
    <tr>
      <td>52</td>
      <td>[bit, people, time, buddy]</td>
      <td>[love]</td>
      <td>1.000000</td>
      <td>26.033292</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>buddy</td>
    </tr>
    <tr>
      <td>53</td>
      <td>[open, people, friend, buddy]</td>
      <td>[like]</td>
      <td>1.000000</td>
      <td>25.514199</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>buddy</td>
    </tr>
    <tr>
      <td>54</td>
      <td>[people, love, send, friend, buddy]</td>
      <td>[like]</td>
      <td>1.000000</td>
      <td>25.514199</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>buddy</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word BUDDY:**

**Straight**
* love
* like
* friend
* chat
* read
* open
* game
* time

**Homosexual**
* fitness
* gym

#### LOVE:


```python
df_90[df_90.word=='love']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>220</td>
      <td>[read, bit, old, post, time]</td>
      <td>[love]</td>
      <td>1.0</td>
      <td>26.033292</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>221</td>
      <td>[bit, game, old, friend, time]</td>
      <td>[love]</td>
      <td>1.0</td>
      <td>26.033292</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>222</td>
      <td>[mind, open, feel, like, friend, time, talk]</td>
      <td>[love]</td>
      <td>1.0</td>
      <td>26.033292</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>223</td>
      <td>[bit, game, post, friend, chat]</td>
      <td>[love]</td>
      <td>1.0</td>
      <td>26.033292</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>224</td>
      <td>[mind, bit, open, people]</td>
      <td>[love]</td>
      <td>1.0</td>
      <td>26.033292</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>230</td>
      <td>[mind, bit, free, love, like]</td>
      <td>[time]</td>
      <td>1.0</td>
      <td>16.743061</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>231</td>
      <td>[com, read, love, like, send]</td>
      <td>[open]</td>
      <td>1.0</td>
      <td>43.000000</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>232</td>
      <td>[read, bit, people, post, love, time]</td>
      <td>[feel]</td>
      <td>1.0</td>
      <td>34.554828</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>233</td>
      <td>[open, feel, post, love, friend]</td>
      <td>[time]</td>
      <td>1.0</td>
      <td>16.743061</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>234</td>
      <td>[bit, free, love, send]</td>
      <td>[time]</td>
      <td>1.0</td>
      <td>16.743061</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>235</td>
      <td>[term, love]</td>
      <td>[long]</td>
      <td>1.0</td>
      <td>44.465116</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>236</td>
      <td>[height, great, love]</td>
      <td>[interested]</td>
      <td>1.0</td>
      <td>14.215613</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>237</td>
      <td>[fair, love]</td>
      <td>[skin]</td>
      <td>1.0</td>
      <td>95.600000</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>238</td>
      <td>[bod, love]</td>
      <td>[dad]</td>
      <td>1.0</td>
      <td>127.466667</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
    <tr>
      <td>239</td>
      <td>[love, work, chat]</td>
      <td>[talk]</td>
      <td>1.0</td>
      <td>9.984334</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>love</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word LOVE:**

**Straight**
* game
* chat
* feel
* read
* old
* time
* friend
* like

**Homosexual**
* long
* term
* height
* dad
* bod
* interested
* fair
* skin

Given the results of the association rule mining on the words **buddy** and **love**, there is a **distinct difference** between the words associated to these against **homosexual and straight** redditors.

Particularly, **straight** users tend to associate ‘buddy’ and ‘love’ with closely similar words such as: **friend, chat, read, and time**. 

On the other hand, **homosexual** users associate the word ‘buddy’ with **fitness and gym** whereas the words about **physical appearance** are distinctly associated with the word ‘love’. 

Interestingly, this suggests that **homosexual users tend to be more straightforward and upfront** with stating their preferences and persona features when it comes to conversing about ‘love’ and **straight users would discuss the topic more casually**, incorporating the topics such as friendship, time, and reading.

#### PLACE:


```python
df_90[df_90.word=='place']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>265</td>
      <td>[provide, big]</td>
      <td>[place]</td>
      <td>1.000000</td>
      <td>22.104046</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>266</td>
      <td>[provide, viber, willing]</td>
      <td>[place]</td>
      <td>1.000000</td>
      <td>22.104046</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>267</td>
      <td>[provide, sweet]</td>
      <td>[place]</td>
      <td>1.000000</td>
      <td>22.104046</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>268</td>
      <td>[marikina, willing]</td>
      <td>[place]</td>
      <td>1.000000</td>
      <td>22.104046</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>269</td>
      <td>[provide, tg]</td>
      <td>[place]</td>
      <td>1.000000</td>
      <td>22.104046</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>270</td>
      <td>[imgur, place]</td>
      <td>[com]</td>
      <td>1.000000</td>
      <td>268.955414</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>271</td>
      <td>[feel, place, chat]</td>
      <td>[time]</td>
      <td>0.921569</td>
      <td>15.429880</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>272</td>
      <td>[lot, open, place]</td>
      <td>[post]</td>
      <td>0.918367</td>
      <td>27.659757</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>273</td>
      <td>[place, post, love, like]</td>
      <td>[time]</td>
      <td>0.916667</td>
      <td>15.347806</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>274</td>
      <td>[mind, place, chat]</td>
      <td>[time]</td>
      <td>0.916667</td>
      <td>15.347806</td>
      <td>Straight</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>275</td>
      <td>[light, place]</td>
      <td>[bdsm]</td>
      <td>1.000000</td>
      <td>191.200000</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>276</td>
      <td>[bdsm, place]</td>
      <td>[light]</td>
      <td>1.000000</td>
      <td>159.333333</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>277</td>
      <td>[obese, place]</td>
      <td>[viber]</td>
      <td>1.000000</td>
      <td>59.750000</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>278</td>
      <td>[big, experience, place]</td>
      <td>[willing]</td>
      <td>1.000000</td>
      <td>25.664430</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
    <tr>
      <td>279</td>
      <td>[deed, place]</td>
      <td>[willing]</td>
      <td>1.000000</td>
      <td>25.664430</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>place</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word PLACE:**

**Straight**
* feel
* chat
* open
* lot
* love
* time
* post

**Homosexual**
* marikina
* willing
* provide
* big
* sweet
* tg
* bdsm
* obese

It is notable that based on the results, **Marikina** is often associated by **homosexuals** to the word **place**. This may be attributed to the fact that the **Metro Manila Pride March**, an annual gathering of the LGBTQI+ community is **held and hosted by the Marikina City Government**. 

#### SEX:


```python
df_90[df_90.word=='sex']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>205</td>
      <td>[decent, safe]</td>
      <td>[sex]</td>
      <td>1.0</td>
      <td>29.190840</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>206</td>
      <td>[safe, body, clean, see]</td>
      <td>[sex]</td>
      <td>1.0</td>
      <td>29.190840</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>207</td>
      <td>[safe, type, curious]</td>
      <td>[sex]</td>
      <td>1.0</td>
      <td>29.190840</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>208</td>
      <td>[decent, safe, average, good]</td>
      <td>[sex]</td>
      <td>1.0</td>
      <td>29.190840</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>209</td>
      <td>[safe, body, see]</td>
      <td>[sex]</td>
      <td>1.0</td>
      <td>29.190840</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>215</td>
      <td>[safe, body, sex, good]</td>
      <td>[type]</td>
      <td>1.0</td>
      <td>46.072289</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>216</td>
      <td>[safe, coffee, sex]</td>
      <td>[type]</td>
      <td>1.0</td>
      <td>46.072289</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>217</td>
      <td>[safe, type, sex, curious]</td>
      <td>[body]</td>
      <td>1.0</td>
      <td>40.680851</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>218</td>
      <td>[clean, pic, sex]</td>
      <td>[body]</td>
      <td>1.0</td>
      <td>40.680851</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
    <tr>
      <td>219</td>
      <td>[decent, safe, average, sex]</td>
      <td>[good]</td>
      <td>1.0</td>
      <td>16.412017</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>sex</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word SEX:**

**Homosexual**
* decent
* safe
* body
* clean
* type
* curious
* drink

Given the results above, it is keen to note that **homosexuals** often associate the word **sex** with words such as: **safe, clean, curious**, etc. This results show that homosexual redditors are **keen on associating sexual activities with safety and cleanliness** – an idea often contradictory to the preconceived prejudice towards the LGBTQI+ community.

#### GIRL:


```python
df_90[df_90.word=='girl']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>125</td>
      <td>[boyfriend, threesome, interested]</td>
      <td>[girl]</td>
      <td>1.0</td>
      <td>11.247059</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>126</td>
      <td>[boyfriend, clean, interested]</td>
      <td>[girl]</td>
      <td>1.0</td>
      <td>11.247059</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>127</td>
      <td>[boyfriend, dm, interested]</td>
      <td>[girl]</td>
      <td>1.0</td>
      <td>11.247059</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>128</td>
      <td>[eat, explore]</td>
      <td>[girl]</td>
      <td>1.0</td>
      <td>11.247059</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>129</td>
      <td>[boyfriend, dm]</td>
      <td>[girl]</td>
      <td>1.0</td>
      <td>11.247059</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>135</td>
      <td>[boyfriend, clean, girl]</td>
      <td>[interested]</td>
      <td>1.0</td>
      <td>14.215613</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>136</td>
      <td>[hope, send, girl]</td>
      <td>[talk]</td>
      <td>1.0</td>
      <td>9.984334</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>137</td>
      <td>[could, interested, girl]</td>
      <td>[make]</td>
      <td>1.0</td>
      <td>21.604520</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>138</td>
      <td>[boyfriend, dm, girl]</td>
      <td>[interested]</td>
      <td>1.0</td>
      <td>14.215613</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>girl</td>
    </tr>
    <tr>
      <td>139</td>
      <td>[old, pm, girl]</td>
      <td>[year]</td>
      <td>1.0</td>
      <td>33.840708</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>girl</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word GIRL:**

**Homosexual**
* boyfriend
* threesome
* interested
* clean
* explore
* eat

Users tagged as LGBTQI+ may not necessarily identify as such, since the frequent occurrence of the words **curious** and **explore** in their posts suggests that the platform is used to **explore bisexual curiosities**. This is consistent with our assumption that **anonymity brings out sentiments that otherwise would not surface** in other more visible platforms.

#### WATCH:


```python
df_90[df_90.word=='watch']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>285</td>
      <td>[ft, height]</td>
      <td>[watch]</td>
      <td>1.0</td>
      <td>25.157895</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>watch</td>
    </tr>
    <tr>
      <td>295</td>
      <td>[finger, watch]</td>
      <td>[threesome]</td>
      <td>1.0</td>
      <td>20.232804</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>watch</td>
    </tr>
    <tr>
      <td>296</td>
      <td>[watch, work, talk]</td>
      <td>[movie]</td>
      <td>0.9</td>
      <td>34.763636</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>watch</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word WATCH:**

**Homosexual**
* height
* finger
* threesome
* work
* talk
* movie

Note that the words **sex, girl and watch** only have association rules with confidence of at least 90% on posts by **homosexual** redditors.

### 3. Association rules with words that pertain to certain cities

#### QC:


```python
df_90[df_90.word=='qc']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>15</td>
      <td>[provide, qc]</td>
      <td>[viber]</td>
      <td>1.0</td>
      <td>59.750</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>qc</td>
    </tr>
    <tr>
      <td>16</td>
      <td>[sweet, qc]</td>
      <td>[viber]</td>
      <td>0.9</td>
      <td>53.775</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>qc</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word QC:**

**Homosexual**
* provide
* sweet
* viber

#### MANILA:


```python
df_90[df_90.word=='manila']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25</td>
      <td>[metro, telegram]</td>
      <td>[manila]</td>
      <td>0.909091</td>
      <td>10.194615</td>
      <td>Homosexual</td>
      <td>Consequent</td>
      <td>manila</td>
    </tr>
    <tr>
      <td>35</td>
      <td>[chat, telegram, manila]</td>
      <td>[talk]</td>
      <td>1.000000</td>
      <td>9.984334</td>
      <td>Homosexual</td>
      <td>Antecedent</td>
      <td>manila</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word MANILA:**

**Homosexual**
* telegram
* chat
* talk

#### MAKATI:


```python
df_90[df_90.word=='makati']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
      <th>lift</th>
      <th>so</th>
      <th>from</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>80</td>
      <td>[dominant, taft]</td>
      <td>[makati]</td>
      <td>0.977778</td>
      <td>13.195156</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>makati</td>
    </tr>
    <tr>
      <td>81</td>
      <td>[dominant, pasay]</td>
      <td>[makati]</td>
      <td>0.956522</td>
      <td>12.908305</td>
      <td>Straight</td>
      <td>Consequent</td>
      <td>makati</td>
    </tr>
  </tbody>
</table>
</div>



**Association with the word MAKATI:**

**Homosexual**
* dominant
* taft
* pasay


Interestingly, the words displayed above (**qc, manila and makati**) only have association rules with confidence of at least 90% on posts by **homosexual** redditors.

## V. Conclusion

Based on our findings, we saw that Filipino Redditors more often speak of similar things but in different contexts. For example, when heterosexuals talk about “buddy” they mean a friend they can chat or spend time with. On the other hand, those from the LGBT community often refer to a “buddy” as someone who they go to the gym with or someone who engages in the same fitness routine. 

Another insight that we got from our study is that Redditors may explicitly be looking for similar things but implicitly mean different things. For example, when heterosexuals say “fun” some might just be looking for people who are interesting and they could spend time with. For the LGBT, “fun” meant having good, clean sex.  

Moreover, in the Philippine R4R, depending on the area, Redditors differ in what they search for. For example, LGBT in Manila just seek people to chat and talk with, while in Quezon City, they look for those who are sweet and provide. On the other hand, in Makati heterosexuals look for dominant companions. 

Overall, we see that the phr4r community is vibrant and diverse, with a language that is different from the usual context of normal conversation. 

Some possible applications for our study include the (1) identification of possible deviant behavior, such as those who seek minors for companionship (pedophiles), which could aid the authorities in investigating them; (2) the tracking of sexual health through monitoring of online activities. This can be useful for the Department of Health especially on how highly communicable sexually transmitted diseases such as AIDS spread in certain geographic areas; (3) gain marketing insight for use in different products such as cellular phones, coffee shops, etc.; (4) creation of a mobile app for date matching whose primary feature is anonymity. 

## VI. Acknowledgement
Special thanks to Professor Christian Alis, Professor Madhavi Devaraj, and Professor Eduardo David of the Asian Institute of Management (Manila, Philippines) for their guidance and support.

## VII. References

https://www.redditinc.com/ 

https://www.reddit.com/r/r4r/comments/6nwgck/meta_welcome_to_rr4r_please_read_this_before/ 

Curlew, A. E. (2019). Undisciplined Performativity: A Sociological Approach to Anonymity. Social Media+ Society, 5(1), 2056305119829843. 

Fournier-Viger, P., Lin, J. C. W., Vo, B., Chi, T. T., Zhang, J., & Le, H. B. A survey of itemset mining. WIREs Data Min. Knowl. Discov. e1207 (2017). 

Aggarwal, C. C. (2016). Recommender systems (pp. 1-28). Cham: Springer International Publishing. 

Arora, J., Bhalla, N., Rao, S. (2013). A Review on Association Rule Mining Algorithms, IJIRCCE International Journal of Innovative Research in Computer and Communication Engineering, Vol. 1, Issue 5, July 2013. 

Black, S. W., Kaminsky, G., Hudson, A., Owen, J., & Fincham, F. (2019). A Short-Term Longitudinal Investigation of Hookups and Holistic Outcomes Among College Students. Archives of sexual behavior, 1-17. 

Barcz, M., Gryz, J., & Wierzbicki, A. (2019). The Logical Structure of Intentional Anonymity. Diametros, (60), 1-17. 


```python

```
