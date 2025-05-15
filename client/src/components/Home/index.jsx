// Home.jsx
import React, { useState, useEffect, useRef } from 'react';
import {Link } from "react-router-dom"
import pc from "../pages/ligand.png";

const Home = ({ setActiveSection }) => {
  const [animateHeader, setAnimateHeader] = useState(false);
  const contactRef = useRef(null);

  useEffect(() => {
    setAnimateHeader(true);
  }, []);

  const scrollToContact = () => {
    contactRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Hero Section */}
      <header className={`py-20 px-6 md:py-28 relative overflow-hidden ${animateHeader ? 'animate-fade-in' : ''}`}>
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-900 to-blue-900/30 z-0"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_107%,rgba(59,130,246,0.08)_0%,rgba(10,10,26,0)_70%)] z-0"></div>
        
        <div className="container mx-auto max-w-5xl relative z-10 text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 text-blue-500 transform transition-all duration-1000 ease-out">
            DrugSeek
          </h1>
          <h2 className="text-2xl md:text-3xl font-semibold mb-3 text-teal-400">
            Revolutionizing Healthcare with AI
          </h2>
          <p className="text-lg text-slate-300 mb-8 max-w-2xl mx-auto">
            Advanced AI Solutions for Medical Diagnostics and Drug Discovery
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link to = "/about"><button 
              
              className="px-6 py-3 rounded-full bg-blue-600 hover:bg-blue-700 text-white font-medium transition duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-blue-500/20"
            >
              Learn More
            </button></Link>
            <button 
              onClick={scrollToContact}
              className="px-6 py-3 rounded-full bg-transparent hover:bg-teal-500/10 text-teal-400 border-2 border-teal-500 font-medium transition duration-300 transform hover:scale-105"
            >
              Contact Us
            </button>
          </div>
        </div>
      </header>

      {/* Key Features */}
      <section className="py-16 px-6 bg-slate-800/60">
        <div className="container mx-auto max-w-6xl">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-slate-900/80 p-8 rounded-xl border border-blue-500/20 hover:border-blue-500/50 transition-all duration-300 hover:transform hover:-translate-y-2 group">
              <div className="text-5xl mb-6 text-teal-400 group-hover:scale-110 transition-transform duration-300">🧬</div>
              <h3 className="text-xl font-semibold mb-3 text-blue-400">AI-Powered Diagnostics</h3>
              <p className="text-slate-300">Cutting-edge machine learning algorithms for precise medical imaging analysis and early disease detection.</p>
            </div>
            
            <div className="bg-slate-900/80 p-8 rounded-xl border border-blue-500/20 hover:border-blue-500/50 transition-all duration-300 hover:transform hover:-translate-y-2 group">
              <div className="text-5xl mb-6 text-teal-400 group-hover:scale-110 transition-transform duration-300">🔬</div>
              <h3 className="text-xl font-semibold mb-3 text-blue-400">Drug Discovery Acceleration</h3>
              <p className="text-slate-300">Innovative computational tools that streamline protein interaction analysis and molecular design processes.</p>
            </div>
            
            <div className="bg-slate-900/80 p-8 rounded-xl border border-blue-500/20 hover:border-blue-500/50 transition-all duration-300 hover:transform hover:-translate-y-2 group">
              <div className="text-5xl mb-6 text-teal-400 group-hover:scale-110 transition-transform duration-300">☁️</div>
              <h3 className="text-xl font-semibold mb-3 text-blue-400">Scalable Cloud Infrastructure</h3>
              <p className="text-slate-300">Secure, flexible cloud solutions powered by AWS to support advanced medical research and diagnostics.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Overview */}
      <section className="py-20 px-6 bg-slate-900">
        <div className="container mx-auto max-w-6xl">
          <div className="mb-16 flex flex-col lg:flex-row items-center justify-between gap-10">
            <div className="lg:w-1/2 flex justify-center">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full blur opacity-30 group-hover:opacity-60 transition duration-1000"></div>
                <div className="relative">
                  <img 
                    src={pc} 
                    alt="Molecular visualization" 
                    className="w-full max-w-md rounded-lg transition-all duration-700 ease-in-out transform group-hover:scale-105 group-hover:saturate-150"
                  />
                </div>
              </div>
            </div>
            
            <div className="lg:w-1/2">
              <h2 className="text-3xl font-bold mb-10 text-center lg:text-left text-blue-500">Our Technological Edge</h2>
              
              <div className="space-y-6">
                <div className="bg-slate-800/60 p-6 rounded-lg border-l-4 border-blue-500 hover:shadow-md hover:shadow-blue-500/10 transition duration-300">
                  <h3 className="text-xl font-semibold mb-2 text-teal-400">Generative AI</h3>
                  <p className="text-slate-300">Leveraging state-of-the-art generative models to transform medical research and diagnostics.</p>
                </div>
                
                <div className="bg-slate-800/60 p-6 rounded-lg border-l-4 border-teal-500 hover:shadow-md hover:shadow-teal-500/10 transition duration-300">
                  <h3 className="text-xl font-semibold mb-2 text-teal-400">Machine Learning</h3>
                  <p className="text-slate-300">Advanced ML algorithms that continuously improve diagnostic accuracy and drug discovery processes.</p>
                </div>
                
                <div className="bg-slate-800/60 p-6 rounded-lg border-l-4 border-indigo-500 hover:shadow-md hover:shadow-indigo-500/10 transition duration-300">
                  <h3 className="text-xl font-semibold mb-2 text-teal-400">Cloud Computing</h3>
                  <p className="text-slate-300">Robust AWS infrastructure ensuring secure, scalable, and high-performance computational capabilities.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Impact Metrics */}
      <section className="py-20 px-6 bg-slate-800/70">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold mb-16 text-center text-blue-500">Our Impact</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-gradient-to-b from-slate-900 to-slate-900/90 p-8 rounded-xl text-center shadow-lg hover:shadow-blue-500/10 transition duration-300">
              <h3 className="text-5xl font-bold mb-4 text-teal-400 flex justify-center">
                <span className="relative">
                  <span className="absolute -inset-1 bg-teal-500/20 blur rounded-lg"></span>
                  <span className="relative">99%</span>
                </span>
              </h3>
              <p className="text-slate-300 text-lg">Diagnostic Accuracy Improvement</p>
            </div>
            
            <div className="bg-gradient-to-b from-slate-900 to-slate-900/90 p-8 rounded-xl text-center shadow-lg hover:shadow-blue-500/10 transition duration-300">
              <h3 className="text-5xl font-bold mb-4 text-blue-400 flex justify-center">
                <span className="relative">
                  <span className="absolute -inset-1 bg-blue-500/20 blur rounded-lg"></span>
                  <span className="relative">60%</span>
                </span>
              </h3>
              <p className="text-slate-300 text-lg">Faster Drug Discovery Timeline</p>
            </div>
            
            <div className="bg-gradient-to-b from-slate-900 to-slate-900/90 p-8 rounded-xl text-center shadow-lg hover:shadow-blue-500/10 transition duration-300">
              <h3 className="text-5xl font-bold mb-4 text-indigo-400 flex justify-center">
                <span className="relative">
                  <span className="absolute -inset-1 bg-indigo-500/20 blur rounded-lg"></span>
                  <span className="relative">10+</span>
                </span>
              </h3>
              <p className="text-slate-300 text-lg">Research Collaborations</p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 px-6 relative">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/20 to-teal-900/20 z-0"></div>
        <div className="container mx-auto max-w-4xl relative z-10 text-center">
          <h2 className="text-3xl font-bold mb-4 text-blue-500">Transform Healthcare with AI</h2>
          <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">Join us in pushing the boundaries of medical technology and drug research.</p>
          <button 
            onClick={() => setActiveSection("brain")} 
            className="px-8 py-4 rounded-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold text-lg shadow-lg shadow-blue-500/20 hover:shadow-blue-600/30 transition-all duration-300 transform hover:scale-105"
          >
            Get Started
          </button>
        </div>
      </section>

      {/* Contact Section */}
      <section ref={contactRef} id="contact" className="py-20 px-6 bg-slate-900 relative">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_30%,rgba(59,130,246,0.05)_0%,rgba(10,10,26,0)_70%)] z-0"></div>
        
        <div className="container mx-auto max-w-6xl relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4 text-blue-500">Contact Us</h2>
            <p className="text-slate-300 text-lg max-w-2xl mx-auto">Have questions or want to collaborate? Reach out to our team.</p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
            <div className="bg-slate-800/40 p-8 rounded-xl border border-blue-500/20">
              <h3 className="text-2xl font-semibold mb-6 text-teal-400 text-center">Send Us a Message</h3>
              
              <form className="space-y-6">
                <div>
                  <input 
                    type="text" 
                    placeholder="Your Name" 
                    required
                    className="w-full px-4 py-3 rounded-lg bg-slate-900/70 border border-slate-700 focus:border-teal-500 focus:ring focus:ring-teal-500/20 focus:outline-none text-slate-100 placeholder-slate-400 transition duration-300"
                  />
                </div>
                
                <div>
                  <input 
                    type="email" 
                    placeholder="Your Email" 
                    required
                    className="w-full px-4 py-3 rounded-lg bg-slate-900/70 border border-slate-700 focus:border-teal-500 focus:ring focus:ring-teal-500/20 focus:outline-none text-slate-100 placeholder-slate-400 transition duration-300"
                  />
                </div>
                
                <div>
                  <select 
                    className="w-full px-4 py-3 rounded-lg bg-slate-900/70 border border-slate-700 focus:border-teal-500 focus:ring focus:ring-teal-500/20 focus:outline-none text-slate-100 transition duration-300 appearance-none"
                  >
                    <option value="" className="bg-slate-900">Select Inquiry Type</option>
                    <option value="partnership" className="bg-slate-900">Partnership</option>
                    <option value="research" className="bg-slate-900">Research Collaboration</option>
                    <option value="support" className="bg-slate-900">Technical Support</option>
                    <option value="other" className="bg-slate-900">Other Inquiry</option>
                  </select>
                </div>
                
                <div>
                  <textarea 
                    placeholder="Your Message" 
                    rows="4" 
                    required
                    className="w-full px-4 py-3 rounded-lg bg-slate-900/70 border border-slate-700 focus:border-teal-500 focus:ring focus:ring-teal-500/20 focus:outline-none text-slate-100 placeholder-slate-400 transition duration-300 resize-none"
                  ></textarea>
                </div>
                
                <button 
                  type="submit" 
                  className="w-full px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition duration-300 transform hover:shadow-lg hover:shadow-blue-500/20"
                >
                  Send Message
                </button>
              </form>
            </div>
            
            <div className="flex flex-col justify-between">
              <div className="space-y-6">
                <div className="flex items-start">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center rounded-full bg-blue-500/10 text-teal-400 text-xl mr-4">
                    📍
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-blue-400 mb-2">Address</h4>
                    <p className="text-slate-300">Udaan Block-F, Keshav Memorial Institute Of Technology, Narayanaguda 500029, Telangana, India.</p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center rounded-full bg-blue-500/10 text-teal-400 text-xl mr-4">
                    📞
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-blue-400 mb-2">Phone</h4>
                    <p className="text-slate-300">+91 9030180427</p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center rounded-full bg-blue-500/10 text-teal-400 text-xl mr-4">
                    ✉️
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-blue-400 mb-2">Email</h4>
                    <p className="text-slate-300">drugseek.med@gmail.com</p>
                  </div>
                </div>
              </div>
              
              <div className="mt-10">
                <h4 className="text-lg font-semibold text-blue-400 mb-4">Connect With Us</h4>
                <div className="flex space-x-4">
                  <a href="#" className="h-10 w-10 flex items-center justify-center rounded-full bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white transition duration-300">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M22.23 0H1.77C.8 0 0 .8 0 1.77v20.46C0 23.2.8 24 1.77 24h20.46c.98 0 1.77-.8 1.77-1.77V1.77C24 .8 23.2 0 22.23 0zM7.27 20.1H3.65V9.24h3.62V20.1zM5.47 7.76c-1.15 0-2.08-.93-2.08-2.08 0-1.15.93-2.08 2.08-2.08 1.15 0 2.08.93 2.08 2.08 0 1.15-.93 2.08-2.08 2.08zm14.63 12.34h-3.62v-5.56c0-1.35-.03-3.1-1.9-3.1-1.9 0-2.17 1.48-2.17 3v5.66H8.8V9.24h3.48v1.6h.05c.48-.92 1.67-1.9 3.45-1.9 3.7 0 4.37 2.43 4.37 5.6v5.56z"></path>
                    </svg>
                  </a>
                  <a href="#" className="h-10 w-10 flex items-center justify-center rounded-full bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white transition duration-300">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84"></path>
                    </svg>
                  </a>
                  <a href="#" className="h-10 w-10 flex items-center justify-center rounded-full bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white transition duration-300">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd"></path>
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 bg-slate-800/80 border-t border-blue-900/20">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col items-center">
            <div className="mb-6 text-center">
              <h3 className="text-2xl font-bold text-blue-500 mb-2">DrugSeek</h3>
              <p className="text-slate-400">Revolutionizing Healthcare with AI</p>
            </div>
            
            <div className="flex flex-wrap justify-center gap-6 mb-8">
              <a href="#" className="text-slate-400 hover:text-teal-400 transition duration-300">Home</a>
              <a href="#" className="text-slate-400 hover:text-teal-400 transition duration-300">About</a>
              <a href="#" className="text-slate-400 hover:text-teal-400 transition duration-300">Services</a>
              <a href="#" className="text-slate-400 hover:text-teal-400 transition duration-300">Blog</a>
              <a href="#" className="text-slate-400 hover:text-teal-400 transition duration-300">Privacy Policy</a>
              <a href="#" className="text-slate-400 hover:text-teal-400 transition duration-300">Terms of Service</a>
            </div>
            
            <div className="text-slate-500 text-sm">
              &copy; 2025 DrugSeek AI. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;