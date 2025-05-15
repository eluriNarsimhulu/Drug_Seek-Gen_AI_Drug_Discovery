

import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./index.css";

const MobileVerification = () => {
    const [phone, setPhone] = useState("");
    const [otp, setOtp] = useState("");
    const [otpSent, setOtpSent] = useState(false);
    const [error, setError] = useState("");
    const [isLoaded, setIsLoaded] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsLoaded(true);
        }, 300);
        
        return () => clearTimeout(timer);
    }, []);

    // Send OTP request
    const sendOtp = async () => {
        if (!phone || phone.trim() === "") {
            setError("Please enter a valid phone number");
            return;
        }

        try {
            const { data } = await axios.post("http://localhost:8080/api/send-otp", { phone });
            setOtpSent(true);
            setError("");
            alert("OTP sent successfully!");
        } catch (error) {
            setError(error.response?.data?.message || "Failed to send OTP");
        }
    };

    // Verify OTP request
    const verifyOtp = async () => {
        if (!otp || otp.trim() === "") {
            setError("Please enter the OTP");
            return;
        }

        try {
            const { data } = await axios.post("http://localhost:8080/api/verify-otp", { phone, otp });
            alert("OTP verified successfully!");
            localStorage.setItem("verifiedPhone", phone);
            navigate("/signup"); // Move to Signup Screen
        } catch (error) {
            setError(error.response?.data?.message || "Invalid OTP");
        }
    };

    return (
        <div className="mobile-bg">
            {/* Particle background effect */}
            <div className="particle-container">
                {Array(15).fill().map((_, index) => (
                    <div 
                        key={index}
                        className="particle"
                        style={{
                            left: `${Math.random() * 100}%`,
                            top: `${Math.random() * 100}%`,
                            animationDelay: `${Math.random() * 5}s`,
                            animationDuration: `${5 + Math.random() * 10}s`,
                            width: `${Math.random() * 15 + 5}px`,
                            height: `${Math.random() * 15 + 5}px`,
                        }}
                    />
                ))}
            </div>

            <div className={`verification-card ${isLoaded ? 'show' : ''}`}>
                <div className="card-header">
                    <div className="card-glow"></div>
                    <h2 className="card-title fade-in">Mobile Verification</h2>
                    <div className="card-subtitle fade-in">
                        {!otpSent ? "Enter your phone number to get started" : "Enter the OTP sent to your phone"}
                    </div>
                </div>

                <div className="card-body">
                    {!otpSent ? (
                        <div className="input-group fade-in" style={{animationDelay: '0.3s'}}>
                            <div className="input-icon">📱</div>
                            <input
                                type="text"
                                placeholder="Enter mobile number"
                                value={phone}
                                onChange={(e) => setPhone(e.target.value)}
                                className="input"
                            />
                        </div>
                    ) : (
                        <div className="input-group fade-in" style={{animationDelay: '0.3s'}}>
                            <div className="input-icon">🔒</div>
                            <input
                                type="text"
                                placeholder="Enter OTP"
                                value={otp}
                                onChange={(e) => setOtp(e.target.value)}
                                className="input"
                            />
                        </div>
                    )}

                    <div className="button-wrap fade-in" style={{animationDelay: '0.6s'}}>
                        {!otpSent ? (
                            <button onClick={sendOtp} className="verification-btn">
                                <span>Send OTP</span>
                                <div className="btn-glow"></div>
                            </button>
                        ) : (
                            <button onClick={verifyOtp} className="verification-btn">
                                <span>Verify OTP</span>
                                <div className="btn-glow"></div>
                            </button>
                        )}
                    </div>

                    {error && (
                        <div className="error-container fade-in" style={{animationDelay: '0.2s'}}>
                            <p className="error-msg">{error}</p>
                        </div>
                    )}
                    
                    <div className="back-option fade-in" style={{animationDelay: '0.8s'}}>
                        <button onClick={() => navigate("/")} className="back-btn">
                            Back to Welcome Screen
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MobileVerification;
