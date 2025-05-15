import { useState } from "react";
import axios from "axios";
import Lottie from "lottie-react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff } from "lucide-react";
import loginAnimation from "../pages/login.json";

const Login = () => {
  const [data, setData] = useState({ email: "", password: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = ({ currentTarget: input }) => {
    setData({ ...data, [input.name]: input.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const url = "http://localhost:8080/api/auth";
      const { data: res } = await axios.post(url, data);
      localStorage.setItem("token", res.data);
      navigate("/home");
    } catch (error) {
      if (
        error.response &&
        error.response.status >= 400 &&
        error.response.status <= 500
      ) {
        setError(error.response.data.message);
      }
    }
  };

  return (
    <div className="w-full min-h-screen bg-black flex items-center justify-center p-4">
      <div className="w-full max-w-5xl flex flex-col-reverse md:flex-row rounded-lg shadow-lg overflow-hidden">
        {/* Left Side - Login Form */}
        <div className="w-full md:w-2/3 bg-white flex flex-col items-center justify-center p-6 rounded-b-lg md:rounded-r-none md:rounded-l-lg">
          <form className="w-full max-w-md flex flex-col items-center" onSubmit={handleSubmit}>
            <h1 className="text-3xl font-bold mb-6">Login to Your Account</h1>
            
            <div className="w-full mb-4">
              <input
                type="email"
                placeholder="Email"
                name="email"
                onChange={handleChange}
                value={data.email}
                required
                className="w-full px-4 py-3 bg-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3bb19b]"
              />
            </div>
            
            <div className="w-full mb-4 relative">
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Password"
                name="password"
                onChange={handleChange}
                value={data.password}
                required
                className="w-full px-4 py-3 bg-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3bb19b]"
              />
              <button 
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-3 text-gray-500 hover:text-gray-700"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
            
            {error && (
              <div className="w-full p-3 mb-4 bg-red-500 text-white rounded-lg text-center">
                {error}
              </div>
            )}
            
            <button
              type="submit"
              className="bg-[#3bb19b] text-white py-3 px-6 rounded-full font-bold mt-2 w-40 hover:bg-[#32a08a] transition-colors"
            >
              Login
            </button>
          </form>
        </div>
        
        {/* Right Side - Animation */}
        <div className="w-full md:w-1/3 bg-[#3bb19b] flex flex-col items-center justify-center p-6 rounded-t-lg md:rounded-l-none md:rounded-r-lg">
          <h1 className="text-white text-3xl font-bold mb-4">New Here?</h1>
          <div className="w-full max-w-xs">
            <Lottie animationData={loginAnimation} loop={true} />
          </div>
          <Link to="/mobile-verification">
            <button
              type="button"
              className="bg-white text-[#3bb19b] py-3 px-6 rounded-full font-bold mt-6 w-40 hover:bg-gray-100 transition-colors"
            >
              Register
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;