import { Link, useNavigate, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";

const Main = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const user = localStorage.getItem("token");

  const [menuOpen, setMenuOpen] = useState(false);
  const [activePath, setActivePath] = useState(location.pathname);

  useEffect(() => {
    setActivePath(location.pathname);
    setMenuOpen(false); // auto-close menu on route change
  }, [location]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  if (!user) return null;

  const navLinks = [
    { path: "/home", label: "Home" },
    { path: "/lung", label: "Lung CT Analysis" },
    { path: "/protein", label: "3D Protein Prediction" },
    { path: "/reinforce", label: "Reinforce" },
    { path: "/chem", label: "Chemberta" },
    { path: "/vit", label: "VIT" },
    { path: "/auto", label: "AutoDocking" },
    { path: "/p2s", label: "Protein to Smile" },
    { path: "/lung-3d-visualization", label: "Visual3D" },
    { path: "/about", label: "About" },
    { path: "/faqs", label: "FAQs" },
  ];

  return (
    <nav className="w-full h-[100px] bg-[#3bb19b] flex items-center justify-between relative z-10">
      {/* Brand/Logo */}
      <h1 
        className="text-white text-4xl ml-5 pl-5 md:pl-20 cursor-pointer font-bold" 
        onClick={() => navigate("/home")}
      >
        DrugSeek
      </h1>

      {/* Mobile Menu Icon */}
      <div 
        className="md:hidden text-white mr-10 cursor-pointer text-3xl" 
        onClick={() => setMenuOpen(!menuOpen)}
      >
        {menuOpen ? '✖' : '☰'}
      </div>

      {/* Navigation Links */}
      <div 
        className={`flex md:flex-row flex-col md:static absolute top-[100px] right-0 
                   bg-[#3bb19b] md:bg-transparent w-full md:w-auto md:h-auto 
                   md:flex transition-all duration-300 z-20
                   ${menuOpen ? 'flex' : 'hidden md:flex'}`}
      >
        <div className="flex flex-col md:flex-row md:gap-8 md:mr-10">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`text-white text-base hover:opacity-80 transition-opacity py-2 px-10 md:px-0 md:py-0 md:mt-3 
                        ${activePath === link.path ? 'underline font-bold' : ''}`}
            >
              {link.label}
            </Link>
          ))}
        </div>
        
        {/* Desktop Logout Button */}
        <button 
          className="hidden md:block bg-white text-black font-bold py-3 px-6 rounded-full w-32 ml-8 mr-20 cursor-pointer"
          onClick={handleLogout}
        >
          Logout
        </button>
        
        {/* Mobile Logout Button */}
        <button 
          className="md:hidden bg-white text-black font-bold py-3 px-6 rounded-full w-32 mx-auto my-4 cursor-pointer"
          onClick={handleLogout}
        >
          Logout
        </button>
      </div>
    </nav>
  );
};

export default Main;