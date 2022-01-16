# Some help/tips for Git things!
Here's a copy of some messages I wrote in the Slack about branches and forks in GitHub just for easy access in the future :D

Okay first a quick summary of what they are/are used for so you don’t have to read everything in those links:

- **Branches**: you’ve got some bit of code that you want to test out…but you’re not quite sure that you want it in the main version of the code. So you end up making a branch and then eventually either merging it into the main branch or deleting it if it didn’t work out
- **Forks**: Copy an entire repository so that you can work on your own version. I usually use this to implement a new feature and then you can make a “pull request” on the main repository (asking them to pull in your code)

Now let’s actually make these things happen. First to fork the repo just go to the GitHub page (link in bookmarks at top of the Slack channel) and hit the “Fork” button in the top right. That should copy the entire repo into a new one associated with your GitHub account. To get that onto your laptop you should then open up terminal and go into the folder where you want to put it and run something like

```git clone https://github.com/<YOUR USERNAME>/thermo_winter22.git```

where you need to replace it with your username of course
At this point you don’t need branches but they are a fairly good way for keeping the repo clean so I recommend it. For example in this pset I would make a branch for handling the wall collisions and then eventually merge that into the main branch. To do this you’d need to run the following

```git branch <YOUR BRANCH NAME>```

Then you need to switch to the branch as well by running

```git checkout <YOUR BRANCH NAME>```

(You can also create/switch to a branch in shorthand by running `git checkout -b <NEW BRANCH NAME>` btw). Once you’ve made some commits and are ready to push you would just run `git push origin <BRANCH>` rather than the usual `git push origin main`.
And that should be basically it. I would handle the merging of branches directly through GitHub honestly because I think it is cleaner and can show you how to do that as well if you like.
