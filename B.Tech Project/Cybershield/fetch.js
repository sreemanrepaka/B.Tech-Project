
let lst=[]
document.getElementById("btn").addEventListener("click", function() {
  let tweet=document.getElementById('text').value;
  if (tweet.length > 0) {
    axios.post('http://127.0.0.1:5000/predict', {
    text: tweet
    
  })


    

    .then(
        (response)=>{
          document.getElementById('result').style.display="block";
          if (response.data!=="not bullying"){
            document.getElementById('result').innerHTML ="Cyberbullying detected! Type of cyberbullying: " + response.data;
            document.getElementById('result').style.backgroundColor="red";
            document.getElementById('result').style.color="white";
            alert("Cyberbullying detected! Report the text immediately!")

          }

          else{
            document.getElementById('result').innerHTML = "No cyberbullying detected!";
            document.getElementById('result').style.backgroundColor="green";
            document.getElementById('result').style.color="white";
          }
        }

        )
    .catch(error => console.error(error));

    }

    else{
      document.getElementById('result').innerHTML = "No text entered";
    }
    

});





function scrape() {

  let bully=0;
  let non_bully=0;
  let total=0;


axios.get('http://127.0.0.1:5000/predict')

  .then(function (response) {
    if (response.data.length > 0) {
      
      lst=response.data
      for (let index = 0; index < response.data.length; index++) {
        total++;
        if (response.data[index]=="not bullying"){
          non_bully++;

        }
        else{
          bully++;
        }
 
      }
      document.getElementById('result').innerHTML =`<h3>Tweet Statistics</h3> <br> Click for more info!`


      
      document.getElementById('loader').style.display="none"
      document.getElementById('stats-container').style.display="flex";
      document.getElementById('total').innerHTML =total;
      document.getElementById('bullying').innerHTML =bully;
      document.getElementById('not-bullying').innerHTML =non_bully;
      alert("Cyberbullying detected in "+bully+" tweet(s). Report the tweets immediately!")

      
    }

   
  })

  .catch(
    error => console.log(error)
  )
  
  document.getElementById('text_area').style.display="none"

  document.getElementById('loader').style.display="block"
  
  document.getElementById('result').innerHTML = "Scraping tweets...";

  setTimeout(() => {
    document.getElementById('result').innerHTML = "Calling API...";
  }, 4000);

  setTimeout(() => {
    document.getElementById('result').innerHTML = "Fetching results...";
  }, 7000);

  
};

document.getElementById("stats-container").addEventListener("click", function() {
axios.get('http://127.0.0.1:5000/tweets')


  .then(function (response) {
    let res=[]
    if (response.data.length > 0) {



          
      res = response.data.reduce((obj,key,index) => {
        obj[key]=lst[index];
        return obj;
      },{});
      
      
      let output = '';

      

  Object.entries(res).forEach(([key, value]) => {
    if (typeof value === 'string' && !value.includes('not bullying')) {
      output += `${key}: ${value}<br>`;
    }
  });




  document.getElementById('tweets').innerHTML =`<h4>Tweets classified as bullying:</h4> ${output}`;



      
    
    }


  })
  
 .catch(
    error => console.log(error)
  )
 })


 const audioFileInput = document.getElementById('audio-file');
 const submitButton = document.getElementById('submit_audio');

 submitButton.addEventListener('click', () => {
  
   const selectedFile = audioFileInput.files[0];
   if (!selectedFile) {
      alert("Please select an image to upload.");
     
   }

   const formData = new FormData();
formData.append('audio', selectedFile, 'my-audio-file.mp3');

axios.post("http://127.0.0.1:5000/transcribe", formData, {
  headers: {
    'Content-Type': 'multipart/form-data'
  }
  
})

.then(response => {
  document.getElementById("transcribe_btn").style.display = "block";
  document.getElementById('loader').style.display="none"
  console.log(response.data);
  document.getElementById("text").value = response.data.text
  if (response.data.loudness>70){
    document.getElementById("result").innerHTML = `<h4><b>Sound Levels: HIGH </b><br> ${Math.round(response.data.loudness)} Decibels</h4>`
    document.getElementById("result").style.color = "red"
    alert("HIGH SOUND LEVELS DETECTED! Can be classified as potential cyberbullying!")

  }

  else{
    document.getElementById("result").innerHTML = "<h4><b>Sound Levels: LOW<b></h4>"
    document.getElementById("result").style.color = "green"

  }

    
  
}).catch(error => {
  console.log(error);
});



document.getElementById('loader').style.display="block"

  

 });
 

document.getElementById("btn3").addEventListener("click", function(e){
  e.preventDefault();
  var dropdown = document.getElementById("input_mode");
  var selectedValue = dropdown.options[dropdown.selectedIndex].value;
  document.getElementById("select_mode").style.display = "none";

  if (selectedValue == "text"){
    document.getElementById("text_area").style.display = "block";

  }

  else if (selectedValue == "audio"){
    document.getElementById("audio_area").style.display = "block";


  }

  else if (selectedValue == "image"){

    document.getElementById("image_area").style.display = "block";
    
  }

  else{
    scrape();
    
  }
  


})

document.getElementById("transcribe_btn").addEventListener("click", function(e){
  document.getElementById("audio_area").style.display = "none";
  document.getElementById("text_area").style.display = "block";
  

})


document.getElementById("submit_image").addEventListener("click", function () {
  // Get the selected image file
  var imageInput = document.getElementById("image-file");
  var imageFile = imageInput.files[0];

  if (imageFile) {
      // Create a FormData object to send the image data
      var formData = new FormData();
      formData.append("image", imageFile);


      // Make a POST request to the API using Axios
      axios.post("http://127.0.0.1:5000/extract_text", formData)
          .then(function (response) {
            document.getElementById('loader').style.display="none"
            document.getElementById('image_area').style.display="none"
            document.getElementById('text_area').style.display="block"
            document.getElementById('text').value = response.data.text
 
          })
          .catch(function (error) {
              console.error("Error:", error);
          });
          document.getElementById('loader').style.display="block"
  } else {
      alert("Please select an image to upload.");
  }
});

document.getElementById('logo').addEventListener('click', function(){
  document.getElementById('select_mode').style.display="block";
  document.getElementById('text_area').style.display="none";
  document.getElementById('image_area').style.display="none";
  document.getElementById('audio_area').style.display="none";
  document.getElementById('loader').style.display="none";
  document.getElementById('sound').style.display="none";
  document.getElementById('result').style.display="none";
  document.getElementById('text').value="";

})

const dropZone = document.getElementById('drop-zone');

dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleDroppedFiles(e.dataTransfer.files);
});

function handleDroppedFiles(files) {
  for (const file of files) {
    if (file.type.startsWith('image/')) {
      var formData = new FormData();
      formData.append("image", file);


      // Make a POST request to the API using Axios
      axios.post("http://127.0.0.1:5000/extract_text", formData)
          .then(function (response) {
            document.getElementById('loader').style.display="none"
            document.getElementById('image_area').style.display="none"
            document.getElementById('text_area').style.display="block"
            document.getElementById('text').value = response.data.text
 
          })
          .catch(function (error) {
              console.error("Error:", error);
          });

          document.getElementById('loader').style.display="block"

      
    }
  }
}