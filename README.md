# fearclassifier
Crowd fear classifier on FMV

#Introduction 
Development of a prototype for collective fear detection on YouTube videos. 

#Data
AffectNet database, obtained with permission from Ali Mollahosseini, Behzad Hasani, and Mohammad H. Mahoor, “AffectNet: A New Database for Facial Expression, Valence, and Arousal Computation in the Wild”, IEEE Transactions on Affective Computing, 2017. URL - http://mohammadmahoor.com/affectnet/

<h2>Preprocessing</h2>
<h2>Data Classification</h2>
- Fear = 6,398 images
- Neutral = randomly split from sample into 6,398 images 

<h1>Classifier</h1>
PyTorch simple CNN, architecture: 

conv2d + relu + conv2d + relu + avg_pool + conv2d + relu + avg_pool + conv2d + relu + FC 

<h1>Evaluation</h1>

<h1>Mean expression calculations</h1>

<h1>Demo FMV</h1>
