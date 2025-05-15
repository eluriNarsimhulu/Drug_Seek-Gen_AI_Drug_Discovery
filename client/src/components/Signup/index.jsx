import { useState, useEffect } from "react";
import axios from "axios";
import Lottie from 'lottie-react';
import { Link } from "react-router-dom";
import signup from "../pages/signup.json";
import { Eye, EyeOff } from "lucide-react";

const Signup = () => {
    const [data, setData] = useState({
        firstName: "",
        lastName: "",
        email: "",
        password: "",
        phone: localStorage.getItem("verifiedPhone") || "",
    });
    const [error, setError] = useState("");
    const [msg, setMsg] = useState("");
    const [showPassword, setShowPassword] = useState(false);

    useEffect(() => {
        // Pre-fill phone number from localStorage
        const verifiedPhone = localStorage.getItem("verifiedPhone");
        if (verifiedPhone) {
            setData((prev) => ({ ...prev, phone: verifiedPhone }));
        }
    }, []);

    const handleChange = ({ currentTarget: input }) => {
        setData({ ...data, [input.name]: input.value });
    };

    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!data.phone) {
            setError("Phone number verification required.");
            return;
        }

        try {
            const url = "http://localhost:8080/api/users/";
            const { data: res } = await axios.post(url, data);
            setMsg(res.message);
        } catch (error) {
            if (error.response && error.response.status >= 400 && error.response.status <= 500) {
                setError(error.response.data.message);
            }
        }
    };

    return (
        <div className="w-full min-h-screen bg-black flex items-center justify-center p-4">
            <div className="w-full max-w-5xl flex flex-col md:flex-row rounded-lg shadow-lg overflow-hidden">
                {/* Left Side */}
                <div className="w-full md:w-1/3 bg-[#3bb19b] flex flex-col items-center justify-center p-6 rounded-t-lg md:rounded-tr-none md:rounded-l-lg">
                    <h1 className="text-white text-3xl font-bold mb-4">Welcome Back</h1>
                    <div className="w-full max-w-xs">
                        <Lottie animationData={signup} />
                    </div>
                    <Link to="/login">
                        <button type="button" className="bg-white text-[#3bb19b] py-3 px-6 rounded-full font-bold mt-6 w-40 hover:bg-gray-100 transition-colors">
                            Login
                        </button>
                    </Link>
                </div>

                {/* Right Side */}
                <div className="w-full md:w-2/3 bg-white flex flex-col items-center justify-center p-6 rounded-b-lg md:rounded-bl-none md:rounded-r-lg">
                    <form className="w-full max-w-md flex flex-col items-center" onSubmit={handleSubmit}>
                        <h1 className="text-3xl font-bold mb-6">Create Account</h1>
                        
                        <div className="w-full mb-4">
                            <input 
                                type="text" 
                                placeholder="First Name" 
                                name="firstName" 
                                onChange={handleChange} 
                                value={data.firstName} 
                                required 
                                className="w-full px-4 py-3 bg-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3bb19b]" 
                            />
                        </div>
                        
                        <div className="w-full mb-4">
                            <input 
                                type="text" 
                                placeholder="Last Name" 
                                name="lastName" 
                                onChange={handleChange} 
                                value={data.lastName} 
                                required 
                                className="w-full px-4 py-3 bg-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#3bb19b]" 
                            />
                        </div>
                        
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
                                onClick={togglePasswordVisibility} 
                                className="absolute right-3 top-3 text-gray-500 hover:text-gray-700"
                            >
                                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                            </button>
                        </div>
                        
                        <div className="w-full mb-4">
                            <input 
                                type="text" 
                                value={data.phone} 
                                disabled 
                                className="w-full px-4 py-3 bg-gray-100 rounded-lg text-gray-600 cursor-not-allowed" 
                            />
                        </div>
                        
                        {error && (
                            <div className="w-full p-3 mb-4 bg-red-500 text-white rounded-lg text-center">
                                {error}
                            </div>
                        )}
                        
                        {msg && (
                            <div className="w-full p-3 mb-4 bg-green-500 text-white rounded-lg text-center">
                                {msg}
                            </div>
                        )}
                        
                        <button 
                            type="submit" 
                            className="bg-[#3bb19b] text-white py-3 px-6 rounded-full font-bold mt-2 w-40 hover:bg-[#32a08a] transition-colors"
                        >
                            Register
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Signup;