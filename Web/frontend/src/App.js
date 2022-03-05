import './App.css';
import './Components/Home/home.js';
import {
  Routes,
  Route
} from "react-router-dom";
import Home from './Components/Home/home.js';
import Detect from './Components/Detect/detect.js';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="detect" element={<Detect />} />
    </Routes>
  );
}

export default App;
