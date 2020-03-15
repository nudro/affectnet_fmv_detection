# fearclassifier
Crowd fear classifier on FMV

#Introduction 
Development of a prototype for collective fear detection on YouTube videos. 

#Data
AffectNet database, obtained with permission from Ali Mollahosseini, Behzad Hasani, and Mohammad H. Mahoor, “AffectNet: A New Database for Facial Expression, Valence, and Arousal Computation in the Wild”, IEEE Transactions on Affective Computing, 2017. URL - http://mohammadmahoor.com/affectnet/

<h2>Preprocessing</h2>
added soon

<h2>Data Classification</h2>
- Fear = 6,398 images
- Neutral = randomly split from sample into 6,398 images 

<h2>Classifier</h2>

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 184 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


<h2>Evaluation</h2>
The current model is 71% on validation, 78% on test.

<h2>Mean expression calculations</h2>
Takes the mean value of the fear probability for all faces on the given frame, saves to outfile.

<h2>Demo FMV</h2>
<a href="https://vimeo.com/397697090">https://vimeo.com/397697090</a>

