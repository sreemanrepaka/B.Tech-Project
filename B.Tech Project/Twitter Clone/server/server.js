const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
const app = express()
const Tweet = require('./models/productModel')

app.use(express.json())
app.use(cors())

mongoose.connect("mongodb://localhost:27017/qwitter")
.then(()=>{
    console.log("connected to database")
})

.catch((error)=>{
    console.log(error)
})

//create new tweet
app.post('/', (req,res)=>{
    let tweet = new Tweet({
        text: req.body.text,
        date: req.body.date,
        type:req.body.type,
        url: req.body.url,
        offensive: req.body.offensive
    })

    tweet.save()
    .then(savedTweet => {
        res.status(201).json(savedTweet);
      })
      .catch(error => {
        res.status(500).json({ error: error.message });
      });
})

//show all tweets
app.get('/',(req,res)=>{
    Tweet.find({})
  .then(tweets => {
    
    res.send(tweets)
  })
  .catch(error => {
    console.log(error);
  });



})

//delete a tweet
app.delete('/:id',(req,res)=>{
  const tweetid = req.params.id

  Tweet.findByIdAndDelete(tweetid)

  .then((deletedtweet) =>{
    
    res.send("successfully deleted")
    console.log("successfully deleted")
  })

  .catch(error=>{
    console.log(error.message)
  })
  

})


app.listen(3000, ()=>{
    console.log("server started on port 3000")
})
