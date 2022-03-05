import React, { Component } from 'react';
import axios from 'axios';
import "./detect.css";

class Detect extends Component {

  state = {
    title: '',
    content: '',
    image: null,
    responseImage: null
  };

  handleChange = (e) => {
    this.setState({
      [e.target.id]: e.target.value
    })
  };

  handleImageChange = (e) => {
    this.setState({
      image: e.target.files[0]
    })
  };

  handleSubmit = (e) => {
    e.preventDefault();
    console.log(this.state);
    let form_data = new FormData();
    form_data.append('file', this.state.image, this.state.image.name);
    let url = 'http://127.0.0.1:8000/predict';
    axios.post(url, form_data, {
      headers: {
        'content-type': 'multipart/form-data'
      }
    })
        .then(res => {
          console.log(res.data);
        })
        .catch(err => console.log(err))
  };

  /*renderresult = (result) => {
    let form_data = new FormData();
    let base64string="";
    let contentType="application/json";
    axios({
      method: 'post',
      url: 'http://127.0.0.1:8000/predict',
      data: form_data,
      headers: {
          'Content-Type': 'multipart/form-data'
      },
      responseType: "arraybuffer"
  })
  .then(response => {
      base64string = btoa(String.fromCharCode(...new Uint8Array(response.data)));
      contentType = response.headers['content-type'];
      return base64string;
  })
  .then(base64string => {
      var image = document.getElementById("myImage");
      image.src = "data:" + contentType + ";base64," + base64string;
      let base64ToString = Buffer.from(image.src, "base64").toString()
      this.setState({
        responseImage: base64ToString
      })
      console.log("BaseString",base64ToString);
  })
  .catch(function(response) {
      console.log(response);
  });
  }*/
  
  renderblobresult = (result) => {
    let form_data = new FormData();
    axios({
      method: 'post',
      url: 'http://127.0.0.1:8000/predict',
      data: form_data,
      headers: {
          'Content-Type': 'multipart/form-data'
      },
      responseType: "arraybuffer"
  })
  .then(response => {
      var blob = new Blob([response.data]);
      return blob;
  })
  .then(blob => {
      var image = document.getElementById("myImage");
      var blobURL = URL.createObjectURL(blob);
      image.src = blobURL;
  })
  .catch(function(response) {
      console.log(response);
  });
  }

  render() {
    return (
      <div className="detect-container">
        <br/>
        <br/>
        <h1>Upload the image to be checked</h1>
        <br/>
        <br/>
        <form onSubmit={this.handleSubmit}>
          <p>
            <input style={{paddingLeft:"10vh"}}type="file"
                   id="image"
                   accept="image/png, image/jpeg"  onChange={this.handleImageChange} required/>
          </p>
          <br/>
          <input type="submit"/>
        </form>
        <img id="myImage" src="" alt="renderedimage"  />
      </div>
    );
  }
}

export default Detect;