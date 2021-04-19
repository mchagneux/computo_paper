---
title: "Template for contribution to Computo"
subtitle: "Example based on the myst system"
author: "The Computo team"
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Template for contribution to computo using myst

## Abstract

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur posuere
vestibulum facilisis. Aenean pretium orci augue, quis lobortis libero accumsan
eu. Nam mollis lorem sit amet pellentesque ullamcorper. Curabitur lobortis
libero eget malesuada vestibulum. Nam nec nibh massa. Pellentesque porttitor
cursus tellus. Mauris urna erat, rhoncus sed faucibus sit amet, venenatis eu
ipsum. 

## Introduction

### About this document

This document provides a Myst template for contributions to the **Computo**
Journal {cite}`computo`. We show how `Python` {cite}`perez2011python`, `R`, or `Julia` code can be included.
Note that you can also add equations easily:

$$
\sum x + y = Z
$$

You can also add equations in such a way to be able to reference them later:

```{math}
:label: math
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

See equation {eq}`math` for details.

## Methods


(subsec:this-is-a-subheading)=
### This is a subheading

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur eget ipsum
est. Mauris purus urna, aliquet non interdum quis, tincidunt in tortor.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac
turpis egestas. Vestibulum efficitur eu mi ac consequat. Sed id velit porta
libero facilisis auctor malesuada ut erat. Sed at orci sem. Nunc eget nisi
lobortis, ultrices justo vel, porttitor dolor. Proin ullamcorper dui purus, eu
maximus lorem pharetra non. Etiam ante purus, tempus eget rhoncus eu, rutrum
id ante. Nulla suscipit, risus vel dapibus finibus, metus diam euismod felis,
sit amet consequat quam lectus quis lacus. Sed tristique, urna sit amet
viverra facilisis, est augue auctor lorem, sed varius lectus nibh eget urna.
Aenean a luctus ligula, vitae elementum metus. Etiam varius, leo in iaculis
rutrum, mauris ex fringilla enim, vitae iaculis justo massa id purus. 

```{code-cell} python3
---
tags: [show-output, show-input]
---
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.arange(10))
```

FIXME: Can we reference figures?

(subsec:subheading)=
### This is another subheading

As seen in [section](subsec:this-is-a-subheading), lorem ipsum dolor sit amet,
consectetur adipiscing elit. Cras nec ornare urna. Nullam consectetur
vestibulum lacinia. Aenean sed augue elit. Aenean venenatis orci ut felis
condimentum, sit amet feugiat eros tincidunt. Vestibulum ante ipsum primis in
faucibus orci luctus et ultrices posuere cubilia curae; Integer varius metus
nunc, at molestie metus eleifend et. Maecenas ullamcorper metus at nisl
molestie, ac commodo arcu mollis. Donec felis odio, fermentum lacinia
vestibulum non, elementum eu metus. Donec suscipit aliquam malesuada. Praesent
justo turpis, dignissim ac nulla non, malesuada rutrum nisi.

```{table} My table title
:name: my-table-ref

| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
```

Now we can reference the table in the text (See {ref}`my-table-ref`).


## Discussion

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent aliquam
porttitor rutrum. Donec in sollicitudin risus, ultrices ultricies nisi.
Vestibulum vel turpis eros. Suspendisse pellentesque nunc augue, eget laoreet
dui dictum ut. Mauris pretium lectus ut elit pulvinar, nec accumsan purus
hendrerit. Phasellus hendrerit orci a vestibulum euismod. Nunc vel massa
facilisis, cursus justo nec, suscipit est. 

- This is a list
- With more elements
- It isn't numbered.

But we can also do a numbered list

1. This is my first item
2. This is my second item
3. This is my third item

## Conclusion

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec molestie mollis
urna, vitae convallis turpis placerat vel. Orci varius natoque penatibus et
magnis dis parturient montes, nascetur ridiculus mus. Vestibulum elit nulla,
laoreet a sagittis non, pharetra ac velit. Aliquam volutpat nisl augue, eget
maximus erat cursus eget. Maecenas sed nisi bibendum, sagittis tellus et,
lacinia augue. In lobortis, libero at auctor aliquam, mi leo egestas orci, a
pulvinar mauris magna in nisi. Morbi cursus dictum dignissim. Fusce at ex sit
amet felis vehicula gravida non in sem. Morbi ut condimentum diam. Aliquam
erat volutpat. Fusce id pharetra ante, tincidunt dapibus eros. Curabitur
mattis magna non felis aliquet sagittis. 

```{bibliography}
:style: unsrt
```
