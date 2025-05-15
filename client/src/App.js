// import { Routes, Route, Navigate } from "react-router-dom";
// import Signup from "./components/Signup";
// import Login from "./components/Login";
// import EmailVerify from "./components/EmailVerify";
// import WelcomeScreen from "./components/WelcomeScreen/WelcomeScreen";
// import MobileVerification from "./components/MobileVerification";
// import Home from "./components/Home";
// import Lung from "./components/Lung";
// import About from "./components/About";
// import Faqs from "./components/Faqs";
// import ProteinPrediction from "./components/proteinVisualization/proteinPrediction";
// import Reinforce from "./components/Reinforce/index.tsx";
// import Chembart from "./components/Chembarta";
// import Vit from "./components/VIT_PREDICT";
// import Visualization from "./components/Visual3d/Visualization";
// import ProteinConverter from "./components/ProtienToSmile/ProteinToSMILES";
// import LigandProcessor from "./components/AutoDocking/LigandProcessor";
// import Navbar from "./components/Main"; // Separate nav component
// import "./index.css";

// const ProtectedRoute = ({ children }) => {
//   const user = localStorage.getItem("token");
//   return user ? children : <Navigate to="/login" />;
// };

// function App() {
//   return (
//     <>
//       <Navbar />
//       <Routes>
//         <Route path="/" element={<Navigate to="/home" />} />
//         <Route path="/signup" element={<Signup />} />
//         <Route path="/login" element={<Login />} />
//         <Route path="/mobile-verification" element={<MobileVerification />} />
//         <Route path="/users/:id/verify/:token" element={<EmailVerify />} />

//         {/* Protected Routes */}
//         <Route path="/home" element={<ProtectedRoute><Home /></ProtectedRoute>} />
//         <Route path="/lung" element={<ProtectedRoute><Lung /></ProtectedRoute>} />
//         <Route path="/protein" element={<ProtectedRoute><ProteinPrediction /></ProtectedRoute>} />
//         <Route path="/reinforce" element={<ProtectedRoute><Reinforce /></ProtectedRoute>} />
//         <Route path="/chem" element={<ProtectedRoute><Chembart /></ProtectedRoute>} />
//         <Route path="/vit" element={<ProtectedRoute><Vit /></ProtectedRoute>} />
//         <Route path="/p2s" element={<ProtectedRoute><ProteinConverter /></ProtectedRoute>} />
//         <Route path="/auto" element={<ProtectedRoute><LigandProcessor /></ProtectedRoute>} />
//         <Route path="/lung-3d-visualization" element={<ProtectedRoute><Visualization /></ProtectedRoute>} />
//         <Route path="/about" element={<ProtectedRoute><About /></ProtectedRoute>} />
//         <Route path="/faqs" element={<ProtectedRoute><Faqs /></ProtectedRoute>} />

//         {/* Catch-all */}
//         <Route path="*" element={<Navigate to="/home" />} />
//       </Routes>
//     </>
//   );
// }

// export default App;


import { Routes, Route, Navigate } from "react-router-dom";
import Signup from "./components/Signup";
import Login from "./components/Login";
import EmailVerify from "./components/EmailVerify";
import WelcomeScreen from "./components/WelcomeScreen/WelcomeScreen";
import MobileVerification from "./components/MobileVerification";
import Home from "./components/Home";
import Lung from "./components/Lung";
import About from "./components/About";
import Faqs from "./components/Faqs";
import ProteinPrediction from "./components/proteinVisualization/proteinPrediction";
import Reinforce from "./components/Reinforce/index.tsx";
import Chembart from "./components/Chembarta";
import Vit from "./components/VIT_PREDICT";
import Visualization from "./components/Visual3d/Visualization";
import ProteinConverter from "./components/ProtienToSmile/ProteinToSMILES";
import LigandProcessor from "./components/AutoDocking/LigandProcessor";
import Navbar from "./components/Main"; // Separate nav component
import "./index.css";

const ProtectedRoute = ({ children }) => {
  const user = localStorage.getItem("token");
  return user ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<WelcomeScreen/>} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />
        <Route path="/mobile-verification" element={<MobileVerification />} />
        <Route path="/users/:id/verify/:token" element={<EmailVerify />} />

        {/* Protected Routes */}
        <Route path="/home" element={<ProtectedRoute><Home /></ProtectedRoute>} />
        <Route path="/lung" element={<ProtectedRoute><Lung /></ProtectedRoute>} />
        <Route path="/protein" element={<ProtectedRoute><ProteinPrediction /></ProtectedRoute>} />
        <Route path="/reinforce" element={<ProtectedRoute><Reinforce /></ProtectedRoute>} />
        <Route path="/chem" element={<ProtectedRoute><Chembart /></ProtectedRoute>} />
        <Route path="/vit" element={<ProtectedRoute><Vit /></ProtectedRoute>} />
        <Route path="/p2s" element={<ProtectedRoute><ProteinConverter /></ProtectedRoute>} />
        <Route path="/auto" element={<ProtectedRoute><LigandProcessor /></ProtectedRoute>} />
        <Route path="/lung-3d-visualization" element={<ProtectedRoute><Visualization /></ProtectedRoute>} />
        <Route path="/about" element={<ProtectedRoute><About /></ProtectedRoute>} />
        <Route path="/faqs" element={<ProtectedRoute><Faqs /></ProtectedRoute>} />

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/home" />} />
      </Routes>
    </>
  );
}

export default App;
