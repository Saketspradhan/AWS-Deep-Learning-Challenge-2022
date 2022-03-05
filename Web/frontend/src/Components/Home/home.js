import React from 'react'
import "./home.css";
import icon from "./icon.png";
import dfe from "./deepfakeexample.jpg";
import Button from "@mui/material/Button";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div id="home-container">
      <div id="home1">
        <h1 className="home-title">Deep-Fake Detector</h1>
        <div id="icon-main-container">
          <img id="icon-main" src={icon} alt="icon" />
        </div>
        <br />
        <div id="home-button-container">
          <Link to="detect">
            <Button
              style={{textDecoration:"None",height: "10vh", width: "auto" }}
              variant="outlined"
              size="large"
            >
              Detect DeepFakes
            </Button>
          </Link>
        </div>
      </div>
      <div id="home2">
        <h1 className="home-title">What is a Deep-Fake?</h1>
        <div id="dfe-container">
          <img id="dfe" src={dfe} alt="dfe" />
        </div>
        <p className="home-subtitle">
          Deepfake technology can create convincing but entirely fictional
          photos from scratch.
        </p>
        <p className="home-subtitle">
          The 21st century’s answer to Photoshopping, deepfakes use a form of
          artificial intelligence called deep learning to make images of fake
          events, hence the name deepfake.{" "}
        </p>
      </div>
    </div>
  );
}

export default Home;
