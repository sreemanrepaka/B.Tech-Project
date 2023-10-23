const mongoose = require('mongoose')
const tweetSchema = mongoose.Schema(
    {
        text: {
            type: String,
            required: true

        },

        date: {
            type: String,
            required: true
        },

        type:{
            type: String
        },

        url:{
            type: String
        },

        offensive:{
            type: String
        }
    }
)

const Tweet = mongoose.model('Tweet', tweetSchema)
module.exports = Tweet

