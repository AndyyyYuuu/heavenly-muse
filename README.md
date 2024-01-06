# Heavenly Muse
> *The Milton-style AI gibberish generator with a fancy name.*

A simple character-based LSTM neural network trained on John Milton's epic poem *Paradise Lost*.  
The project follows this [PyTorch LSTM text generator tutorial](https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/).  
For more on how it works and a postmortem, see [my article](). 

### Try It Out?
**Dependencies:** `torch`, `numpy`.  
Run `textGen/generate.py` with Python3. 
```bash
python3 textGen/generate.py
```
Or, if you couldn't be bothered, take a look at some samples below 👇👇

### Sample Generated Text
7 Epochs, Temperature 0.6, *Paradise Lost* & *Paradise Regain'd* (model 3)
```gibberish
oud deo ware
mf ham spietoe mod sert—tf deg rheos;seeeemogt af susesonn
, whe brid
tf gim fhusi,,whe fryeegf jia leinte ohe bepteff pin santn ,shp gune.oa fas eoeot

oha moae
af sea fhnp
d tho waes tf hin furot,

he wrau nu dos shnod 
fht bont af poa so ns
gsod seolngf rir dooe   uheiswno
af woa hrrd,,rohe sae  of sir frpn
d,the morn nm tob,wernns tpe bei
imf mtv gouns,;the llnn
df hes trent  oheesrin fr mis tuandi;ihoihnuo af ovs shanp ,ihe waaseaf las tonp mrsae shnsipf aesesernus tha iobe
sf dls eurg
;,oho coyp
tf tes wiat
r
geayptne.hfsgus bhre. ,hhe mere
nf gim fooru,rjhe pone ao pir srutd,;bheeownp,af hig woano;
mhe wtncsof mev jooeds bha soao
da mad hteees,bheyfret
mu gos hamod,,oheesor,laf oov seinpr roi iwrp,of ael eiuit, ioe stenssr wed ceeud

hhe tour;ou gis oiadso uoeisrukhwn faa moren
 theesofl
ar aosesioe  
aie airl of peg boanel fhe fobgetf wea soi d 
ohe fon- af tos bowpts tog rrr 
we drs trng
, ohe fone
to oun heig
n,wheshpnt dp fav leand, ahe hrne pf tos cons d
ttoeae
```
--- 
60 Epochs, Temperature 1.2, *Paradise Lost* (model 2)
```gibberish
pfante,ap swgs nftte ssdlhuifonki tf tent ,w enbruce
pf daar  hum awgre if roarteah maadtlmh ygoah
.if hne   tri raareuof hosntebf mor t;vte sunre
of,meynegaub sewkiotf gapcalsocgsazt  peywaime:bfhtaere mhi ecdde,of tenrtroragaae eatotsoor

ef tiiso,hhrisoarelib getrertf,penrteyhoupoard
of gicte tra mmrlnktf lialunwh yhlly,bha pei


wfcloete brrylanrirof sln t wf yiadyeeoe baine,gj snu eeooe dun-esoi hiank of haor  nhennknn

oi beose toi btareeof hendi umosnndessreskaal’dof wiuw  eoe facsiwof terrerwfffel t phl rhawe,bf ewmv fniu pooneoof gunresma gead eup  eral

hf ceesefgi’uraunersr totetetu aiade chinson.e
co leae fciu ceniy of geiderhutsabre,hhetaoik
.ba toaneeth’ybioeeear renneiof hearts ootsau,s
af tant  aae epeceoif recnk tn fonbe afiofonpi:hffmoiseefhe reotessu ciaaeukf delaeskheugiaks
hn dpaseeahfyseadeavi eil eerr heme eihrumesme
of saas ,teiisluredtl laodonihocerae,theuean,e
tf ahsee src wwarelif oianleef aeirdarhe koase

ftgpcn,fhht wonke oh vailyhonrmaa eebhettouhe
of bhws
```

