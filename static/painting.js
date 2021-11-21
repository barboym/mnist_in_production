"use strict";

$(function(){
  init()
  // run()
})

//canvas and painting arguments
let pressed = false;
let canvas;
let context;
let last_loc = 0
let lineWidth=5
let lineColor = "#ACD3ED"

//load model
// async function run(){
//     const MODEL_URL = 'model.json';
//     const model = await tf.loadLayersModel(MODEL_URL);
//     console.log(model.summary());
// }

//load tensoflow model
// const tf = require("@tensorflow/tfjs");
// const tfn = require("@tensorflow/tfjs-node");
// const handler = tfn.io.fileSystem("./model.json");
// const model = tf.loadLayersModel(handler);
// import * as tf from "@tensorflow/tfjs"



//canvas
function init(){
    // Get a reference to the canvas
    canvas = document.getElementById('canvas');
    context = canvas.getContext('2d');

    //line style
    context.fillStyle = lineColor
    context.lineWidth = lineWidth;
    context.lineCap = "round";
    context.strokeStyle = lineColor;
}
//mouse tracker + painter

function get_event_location(e){
  return {x:e.clientX - canvas.offsetLeft, y:e.clientY - canvas.offsetTop}
}

document.onmousedown = function(e) {
  pressed = true;
  let loc = get_event_location(e)
  context.arc(loc.x,loc.y,lineWidth/2,0,3.14*2)
  context.fill()
  last_loc = loc
}
document.onmouseup = function(e) {
  pressed = false;
  last_loc=0
}
document.onmousemove = function(e) {
  if (pressed && last_loc!=0){
    let loc = get_event_location(e)
    context.beginPath()
    context.moveTo(last_loc.x,last_loc.y)
    context.lineTo(loc.x,loc.y)
    context.stroke();
    context.fill()
    last_loc=loc
  }
}


function get_image(){
  let image = new Image();
  image.id = "pic";
  image.src = document.getElementById('canvas').toDataURL();
  return image
}


//button tracker
document.getElementById('download').onclick = function(){
  let image = get_image()
  //to dowload the image
  var link = document.createElement('a');
  link.href = image.src;
  link.download = 'image.jpg';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

document.getElementById('generate').onclick = function(){
  // Generate the image data
  var img= document.getElementById("canvas").toDataURL("image/png");
  img = img.replace(/^data:image\/(png|jpg);base64,/, "")

  // Sending the image data to Server
  $.ajax({
      type: 'POST',
      url: '/upload',              // /new
      data: '{ "imageData" : "' + img + '" }',
      contentType: 'application/json; charset=utf-8',
      dataType: 'json',
      success: function (msg) {
      // On success code
      }
  });
}

//
//   fetch("/model", {method: "POST", headers: {'Content-Type': 'application/json'}, body: {"data":document.getElementById('canvas')}});
// };

document.getElementById('clear').onclick = function(){
  context.fillStyle = '#ffffff'
  context.fillRect(0,0,canvas.width,canvas.height)
};
