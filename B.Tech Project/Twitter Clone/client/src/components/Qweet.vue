<template>
  <div>
    <q-item clickable v-ripple>
        <q-item-section avatar top>
          <q-avatar>
            <img src="sreeman-modified.png" >
          </q-avatar>
        </q-item-section>

        <q-item-section top>
          <q-item-label lines="1">Sreeman Repaka</q-item-label>
          <q-item-label caption lines="2">
           {{ qweet.text }}
          </q-item-label>
        </q-item-section>

        <q-item-section side top>
          {{ qweet.date }}
        </q-item-section>
        <q-icon name="delete" size="md"  class="q-pa-md" @click="deleteQweet(qweet._id)"/>

        
       
      
    <img :src="qweet.url" :class="imageClass" style="max-width: 200px; max-height: 200px;">
  
        


      </q-item>

      <q-banner inline-actions class="text-white bg-red" v-if="isBullying(qweet.type)">
      Cyberbullying detected in the above tweet! 
      Type of Cyberbullying: {{ qweet.type }}
      <template v-slot:action>
        <q-btn flat color="white" label="Delete" @click="deleteQweet(qweet._id)" />
      </template>
    </q-banner>


       
  </div>
    


</template>

<script>
    export default {
        
            props: {
                qweet: {
                    type: Object,
                     
                },
            },
           
            
        
        methods: {
            deleteQweet(date) {
                
                console.log(date)
                this.$emit('delete-qweet',date)
                
            },

            isBullying(type){
              if (type !== "not bullying"){
                return true
              }
              else{
                return false
              }

            }
        },

        computed: {
    imageClass() {
      return {
        'conditional-blur': true, // This class is always applied
        'blurred': this.qweet.offensive === 'Yes', // Apply 'blurred' class conditionally
      };
    }
  },


    }
</script>

<style lang="scss" scoped>


.conditional-blur {
  filter: blur(0); /* Initial, no blur */
}

.blurred {
  filter: blur(25px); /* Apply a blur effect */
}





</style>