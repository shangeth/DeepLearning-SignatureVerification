# DeepLearning-SignatureVerification

I am a Mathematics student and AI Developer, so i will first formulate the best mathematical model for this problem and then talk about the code, technologies..etc.

****
# Mathematical Modelling of Sign Recognition/Verification
****
## Task: 
To verify customer's or anybody's signature , to verify if the signature belongs to the customer.
****
## Problem:
We can use Deep Learning for this problem also BUT the problem is  **We cannot get many datas(signatures) of the customers for the deep learning model **. 
So we need to tune the mathematics of Deep Learning such that even with one signature , our model should be able to verify if the signature belongs to the original customer's signature.
****
## Mathematically modelling the problem:
We need to model the problem such that the model can learn the signature with just one signature data. 
1. So I will use a **convolution network to learn the signature's edges, curves, pattern of curves, .. etc.(pixel level)**
2. Convert the convolution network into a **n dimensional vector**. 
3. So each customer's signature can be represented mathematically using a vector in n dimensional space.
as in the image
![](https://sketch.io/render/sk-cafa99331fce3da945e25508e96275b3.jpeg)

Right Now we don't bother about how to actually get the vectors for each customer's signature. Let's see that in below. 
****
### Comparing the signature vectors for verification:
Now we have Signature Vector for each customer's signature 

![](https://sketch.io/render/sk-e4c3a422960172822a0f4598934c07f7.jpeg)

So we can define a metric for these n dimensional signature vector such that 
* the metric difference between 2 signature vectors are very small if both the signatures are same.
* or the difference is very big if the signatures are different.
****
#### Example in a 3D space, But the actual signature vector will have many dimensions.
![](https://sketch.io/render/sk-dcde7c62f0088c31261cdc4ac1bad1e4.jpeg)


### Metric to verification of signature vectors:
Let's consider S1(signature 1) , S2(signature 2), S3(signaure 3), .....Sn(signature n) ; S's are the signature vectors of each signature
Such that 
* S1 = S2 , 
* S1 != S3 
* ..
* ..
* S1 != Sn.

S1 and S2 are both same signatures. 

So we need a metric d such that 
* d(S1,S2) is small
* d(S1, S3) is very large
* .
* .
* d(S1, Sn) is very large

so, 
d(S1, S2) <= d(S1, S3)    
=>     d(S1, S2) - d(S1, S3) <=0
Here we can notice when we use neural network to fit the above equation , there is a possibility that the network will settle at the signature vector of any signature = 0.
SO to avoid it,  we add a margin A such that 
d(S1, S2) - d(S1, S3)  + A <=0

we difine our loss as L(S1,S2,S3) = d(S1, S2) - d(S1, S3)  + A 
but L is always <= 0 , so we difine 
**L(S1,S2,S3) = max(d(S1, S2) - d(S1, S3)  + A , 0)**

****
#### Datasets
It will actually be better is we can get if not many 3-7 images of signatures , so we try to get some images through customers or we use data augumentation to generate some images.

**L(S1,S2,S3) =sum over datas( max(d(S1, S2) - d(S1, S3)  + A , 0)) **
****

### Alternative model 
So now we have the Signature vectors of every customer. 
Now we have a new signature , we need to verify if this belongs to a specific customer or not. 
We can use logistic regression at the end of the network to classify as 0 if its different or 1 if both are same.


****

## Technical Details
After training the model using python with tensorflow/keras, i have planned to make a web app or software (as per requirement) to get take some signatures with customer id, train it and upload a new sign to verify whose signature it is.
