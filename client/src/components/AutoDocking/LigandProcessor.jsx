import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const LigandProcessor = () => {
  const [ECnumber, setECnumber] = useState('');
  const [ligandId, setLigandId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [pdbContent, setPdbContent] = useState(null);

  const viewerRef = useRef(null);

  const handleProcess = async () => {
    if (!ECnumber.trim() || !ligandId.trim()) {
      setError('Please enter both EC number and ligand ID');
      return;
    }
    
    setLoading(true);
    setError('');
    setPdbContent(null);

    try {
      const response = await axios.post('http://localhost:5000/auto/process', {
        ECnumber,
        LIGAND_ID: ligandId
      });

      if (response.data.pdb_content) {
        setPdbContent(response.data.pdb_content);
      } else {
        setError('No PDB content received.');
      }
    } catch (err) {
      setError('Failed to process. Please check the EC number and ligand ID.');
      console.error(err);
    }

    setLoading(false);
  };

  useEffect(() => {
    if (pdbContent && viewerRef.current) {
      const initViewer = async () => {
        const $3Dmol = await import('3dmol');

        // Ensure the viewer element is visible and sized properly
        viewerRef.current.style.width = '100%';
        viewerRef.current.style.height = '400px';
        viewerRef.current.style.position = 'relative';
        viewerRef.current.innerHTML = ''; // Clear previous

        const viewer = $3Dmol.createViewer(viewerRef.current, {
          backgroundColor: '#1c1c2e', // dark background
        });

        viewer.addModel(pdbContent, 'pdb');
        viewer.setStyle({}, { cartoon: { color: 'spectrum' } });

        if (ligandId) {
          viewer.setStyle({ resn: ligandId.toUpperCase() }, {
            stick: { radius: 0.3, color: 'green' },
            sphere: { radius: 0.5, color: 'green' }
          });
        }

        viewer.zoomTo();
        viewer.render();
      };

      initViewer();
    }
  }, [pdbContent, ligandId]);

  const [molecules, setMolecules] = useState([]);

  useEffect(() => {
    const generateMolecules = () => {
      const newMolecules = [];
      for (let i = 0; i < 15; i++) {
        newMolecules.push({
          id: i,
          left: Math.random() * 100,
          top: Math.random() * 100,
          size: Math.random() * 40 + 20,
          duration: Math.random() * 40 + 20,
          delay: Math.random() * 10
        });
      }
      setMolecules(newMolecules);
    };
    
    generateMolecules();
  }, []);


  return (
    <div className="w-full min-h-screen flex justify-center p-8">
      <div className="absolute inset-0 overflow-hidden z-0">
        {molecules.map((molecule) => (
          <div
            key={molecule.id}
            className="absolute bg-gradient-to-br from-indigo-400/10 to-purple-400/10 rounded-full"
            style={{
              left: `${molecule.left}%`,
              top: `${molecule.top}%`,
              width: `${molecule.size}px`,
              height: `${molecule.size}px`,
              animation: `float ${molecule.duration}s ease-in-out ${molecule.delay}s infinite`,
            }}
          />
        ))}
      </div>
      <div className="w-full max-w-4xl bg-white bg-opacity-95 backdrop-blur-lg rounded-3xl shadow-2xl overflow-hidden">
        
        {/* Header */}
        <header className="bg-gradient-to-r from-blue-900 to-blue-700 text-white p-10 text-center relative border-b border-white border-opacity-10">
          <div className="absolute inset-0 opacity-5 bg-repeat" style={{ 
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E")`
          }}></div>
          <h1 className="text-4xl font-extrabold m-0 leading-tight tracking-tight z-10 relative text-shadow">AutoDock Ligand Processor</h1>
          <div className="text-lg font-normal opacity-80 mt-2 z-10 relative">Vision Transformer (ViT) Model for Molecular Interactions</div>
        </header>
        
        {/* Main Content */}
        <main className="p-10">
          <form className="flex flex-col gap-6" onSubmit={(e) => { e.preventDefault(); handleProcess(); }}>
            <div className="flex flex-col mb-4">
              <label htmlFor="ecnumber" className="font-semibold mb-3 text-blue-900 text-lg">EC Number:</label>
              <div className="relative">
                <input
                  type="text"
                  id="ecnumber"
                  className="w-full py-4 px-5 border-2 border-gray-200 rounded-xl text-lg shadow-sm transition-all focus:outline-none focus:border-blue-900 focus:shadow-md"
                  value={ECnumber}
                  onChange={(e) => setECnumber(e.target.value)}
                  placeholder="e.g. 3.1.1.1"
                  required
                />
              </div>
            </div>
            
            <div className="flex flex-col mb-4">
              <label htmlFor="ligandid" className="font-semibold mb-3 text-blue-900 text-lg">Ligand ID:</label>
              <div className="relative">
                <input
                  type="text"
                  id="ligandid"
                  className="w-full py-4 px-5 border-2 border-gray-200 rounded-xl text-lg shadow-sm transition-all focus:outline-none focus:border-blue-900 focus:shadow-md"
                  value={ligandId}
                  onChange={(e) => setLigandId(e.target.value)}
                  placeholder="e.g. ATP"
                  required
                />
              </div>
            </div>
            
            <button 
              type="submit" 
              className={`flex justify-center items-center gap-2 bg-gradient-to-r from-blue-900 to-blue-700 text-white border-none py-4 px-6 rounded-xl text-lg font-semibold cursor-pointer transition-all shadow-md hover:shadow-lg hover:translate-y-px ${loading ? 'opacity-90' : 'hover:bg-blue-800'} h-14 mt-4 relative overflow-hidden`}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="w-5 h-5 border-3 border-white rounded-full border-t-transparent animate-spin"></span>
                  <span>Processing...</span>
                </>
              ) : (
                'Run AutoDock'
              )}
            </button>
          </form>
          
          {error && (
            <div className="mt-6 p-5 bg-red-50 text-red-700 rounded-xl flex gap-4 items-start border border-red-200 shadow-sm">
              <div className="bg-red-400 text-white w-6 h-6 rounded-full flex items-center justify-center font-bold flex-shrink-0">!</div>
              <div className="text-red-700">
                <strong>Error:</strong> {error}
              </div>
            </div>
          )}
          
          {pdbContent && !error && (
            <div className="mt-10 rounded-2xl overflow-hidden bg-white shadow-lg border border-gray-100 animate-fadeIn">
              <div className="bg-gradient-to-r from-cyan-500 to-blue-500 text-white p-5 flex items-center gap-3">
                <div className="bg-white bg-opacity-30 rounded-full w-8 h-8 flex items-center justify-center font-bold text-xl">✓</div>
                <h2 className="m-0 text-xl font-semibold">3D Interaction Visualization</h2>
              </div>
              <div className="p-6">
                <div className="mb-5">
                  <span className="font-semibold text-sm text-gray-600 uppercase tracking-wider">EC Number:</span>
                  <div className="mt-2 p-3 bg-gray-50 rounded-lg font-mono text-base break-all border border-gray-100 text-blue-800">
                    {ECnumber}
                  </div>
                </div>
                <div className="mb-5">
                  <span className="font-semibold text-sm text-gray-600 uppercase tracking-wider">Ligand ID:</span>
                  <div className="mt-2 p-3 bg-gray-50 rounded-lg font-mono text-base break-all border border-gray-100 text-teal-700 font-semibold">
                    {ligandId}
                  </div>
                </div>
                <div className="w-full h-96 border border-gray-200 rounded-lg overflow-hidden">
                  <div 
                    ref={viewerRef} 
                    className="w-full h-full" 
                  />
                </div>
              </div>
            </div>
          )}
          
          {!pdbContent && !loading && !error && (
            <div className="mt-10 p-8 text-center text-gray-500 border border-gray-200 border-dashed rounded-xl bg-gray-50">
              <p className="text-lg">Enter EC Number and Ligand ID to see 3D visualization.</p>
            </div>
          )}
        </main>
        
        {/* Footer */}
        <footer className="p-6 text-center text-gray-500 border-t border-gray-200 text-sm">
          <div className="opacity-80">Powered by AutoDock & 3DMol.js Technology</div>
        </footer>
      </div>
    </div>
  );
};

export default LigandProcessor;