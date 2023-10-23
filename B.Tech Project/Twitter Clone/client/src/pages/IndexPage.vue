<template>
  <q-page class="q-pa-sm" >

    <q-input bottom-slots v-model="text" label="What's Happening?" counter maxlength="100" :dense="dense" @keyup.enter="addNewQweet">
        <template v-slot:before>
          <q-avatar>
            <img src="/sreeman-modified.png" >
          </q-avatar>
        </template>


        <template v-slot:after>
          <q-btn round dense flat icon="send"  @click="addNewQweet" />
        </template>
      </q-input>

      <q-btn round dense flat icon="photo"  @click="toggle=!toggle" />

      <q-input v-model="url" label="Enter image URL" v-if="toggle"></q-input>

      

      <q-separator />
      <transition-group
  appear
  enter-active-class="animated fadeIn"
  leave-active-class="animated fadeOut"
>
<q-list class="q-pa-sm" v-for="qweet in qweets" :key="qweet.date" >
        <Qweet :qweet="qweet" @delete-qweet="deleteQweet" />
        <q-separator />
      </q-list>
    
  
</transition-group>

  </q-page>
</template>

<script>
import { defineComponent, onMounted } from 'vue'
import Qweet from '../components/Qweet.vue';
import axios from 'axios';



export default defineComponent({


  name: 'IndexPage',
  data() {
    return {
      text: '',
      type: '',
      toggle: false,
      url:'',
      offensive: false,
      qweets:[       
      ]

      
    }
  },
 mounted() {

   axios.get("http://localhost:3000/")
   .then(response=>{
     console.log(response.data)
     this.qweets = response.data
     
   })
   .catch(error=>{
     console.log(error)
   })



  },
  methods: {
    addNewQweet() {
  axios.post("http://127.0.0.1:5000/predict", {
    text: this.text
  })
  .then(response => {
    console.log(response.data);
    this.type = response.data;

    axios.post("http://127.0.0.1:5000/detectnudity",{
      url: this.url
    })

    .then(response => {
      this.offensive = response.data;
      console.log(this.offensive)
      let now = new Date(); // Get the current date and time
      let year = now.getFullYear();
      let month = String(now.getMonth() + 1).padStart(2, '0');
      let day = String(now.getDate()).padStart(2, '0');
      let hours = String(now.getHours()).padStart(2, '0');
      let minutes = String(now.getMinutes()).padStart(2, '0');

      let formattedDate = `${hours}:${minutes} ${day}-${month}-${year} `;


      let newQweet = {
      text: this.text,
      date: formattedDate,
      type: this.type,
      url: this.url,
      offensive: this.offensive
    };

    this.qweets.push(newQweet);


    axios.post('http://localhost:3000/', newQweet)
      .then(response => {
        console.log('Tweet created:', response.data);
        

      })
      .catch(error => {
        console.error('Error creating tweet:', error);
      });

    this.text = '';
    this.url = ''
    
    this.toggle = false;
    
  });

    })
    
},

deleteQweet(id){

      
this.qweets=this.qweets.filter((qweet)=>qweet._id!==id);

axios.delete(`http://localhost:3000/${id}`)

},


  
},
  components: {
    Qweet,
  },
  

})
</script>
