import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from "react-router-dom";
import './home.css';


const Home = ({ setActiveSection }) => {
    const routerNavigate = useNavigate();
    const [animateHeader, setAnimateHeader] = useState(false);
    const [isVisible, setIsVisible]         = useState({});
    const [menuOpen, setMenuOpen]           = useState(false);

    const contactRef    = useRef(null);
    const featuresRef   = useRef(null);
    const demoRef       = useRef(null);
    const techRef       = useRef(null);
    const impactRef     = useRef(null);
    const ctaRef        = useRef(null);

    // ── Safe navigation: works even when prop is missing ──────────────────
    const navigate = (id) => {
        setMenuOpen(false);
        if (typeof setActiveSection === 'function') {
            setActiveSection(id);
        } else {
            window.location.hash = id;
        }
    };

    const openModule = (id) => {
        const routes = {
            lung: "/lung",
            visual3d: "/lung-3d-visualization",
            protein: "/protein",
            autodocking: "/auto",
            vit: "/vit",
            chemberta: "/chem",
            reinforce: "/reinforce",
        };

        if (routes[id]) {
            routerNavigate(routes[id]);
        } else {
            console.warn("Unknown module:", id);
        }
    };


    useEffect(() => {
        setTimeout(() => setAnimateHeader(true), 80);

        const refs = { features: featuresRef, demo: demoRef, technology: techRef, impact: impactRef, cta: ctaRef };
        const obs = new IntersectionObserver(
            (entries) => entries.forEach(e => {
                if (e.isIntersecting) setIsVisible(p => ({ ...p, [e.target.id]: true }));
            }),
            { threshold: 0.1 }
        );
        Object.values(refs).forEach(r => r.current && obs.observe(r.current));
        return () => Object.values(refs).forEach(r => r.current && obs.unobserve(r.current));
    }, []);

    const scrollTo = (ref) => { setMenuOpen(false); ref.current?.scrollIntoView({ behavior: 'smooth' }); };

    // ── Data ───────────────────────────────────────────────────────────────
    const features = [
        // { id: '/brain',       icon: '🧠', title: 'Brain Tumor Detection',        desc: 'Upload MRI scans and get instant AI-powered segmentation of brain tumors with pixel-level precision using our UNETR model.',       tag: 'Segmentation',       color: '#7c5cff', glow: 'rgba(124,92,255,0.22)'  },
        { id: 'lung',        icon: '🫁', title: 'Lung Tumor Segmentation',       desc: 'Analyze CT scans for early-stage lung tumor detection with highlighted overlay visualizations and downloadable results.',             tag: 'CT Analysis',        color: '#2bd9c3', glow: 'rgba(43,217,195,0.22)'  },
        { id: 'visual3d',    icon: '🔭', title: '3D Diagnosis Viewer',           desc: 'Interact with volumetric .nii.gz images in real-time — rotate, zoom, and explore diagnostic data in 3D.',                           tag: 'NIfTI / 3D',         color: '#3a6fff', glow: 'rgba(58,111,255,0.22)'  },
        { id: 'protein',     icon: '🧬', title: 'Protein Structure Prediction',  desc: 'Enter an amino acid sequence and visualize its predicted 3D folded structure using ESMFold-based deep learning.',                   tag: 'Structural Biology', color: '#ff6b6b', glow: 'rgba(255,107,107,0.22)' },
        { id: 'autodocking', icon: '⚗️', title: 'Ligand–Protein Docking',        desc: 'Input an EC number and ligand ID to run AutoDock simulations and visualize 3D molecular binding interactions.',                    tag: 'Molecular Docking',  color: '#f9a825', glow: 'rgba(249,168,37,0.22)'  },
        { id: 'vit',         icon: '🔬', title: 'Molecular Property Classifier', desc: 'Use Vision Transformer (ViT) to classify molecular properties from SMILES strings for rapid drug candidate screening.',             tag: 'SMILES / ViT',       color: '#00e5a0', glow: 'rgba(0,229,160,0.22)'   },
        { id: 'chemberta',   icon: '🧪', title: 'Masked SMILES Predictor',       desc: 'Leverage ChemBERTa transformer to predict masked atoms in SMILES strings with ranked probability scores.',                          tag: 'NLP / Chemistry',    color: '#e040fb', glow: 'rgba(224,64,251,0.22)'  },
        { id: 'reinforce',   icon: '💊', title: 'RL Drug Generation',            desc: 'Generate novel drug-like molecules via reinforcement learning, optimized for target properties and druglikeness scores.',            tag: 'Generative AI',      color: '#ff9a3c', glow: 'rgba(255,154,60,0.22)'  },
    ];

    const impacts = [
        { icon: '📈', value: '99%',   label: 'Diagnostic Accuracy'        },
        { icon: '⚡', value: '60%',   label: 'Faster Drug Discovery'      },
        { icon: '🤝', value: '10+',   label: 'Research Collaborations'    },
        { icon: '🧪', value: '5000+', label: 'Compounds Analyzed'         },
    ];

    return (
        <div className="ds-home">

            {/* ── NAV ───────────────────────────────────────────────── */}
            {/* <nav className="ds-nav">
                <div className="ds-nav__inner">
                    <div className="ds-nav__brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
                        <div className="ds-logo-chip">DS</div>
                        <span>DrugSeek</span>
                    </div>

                    <div className={`ds-nav__links ${menuOpen ? 'ds-nav__links--open' : ''}`}>
                        <button className="ds-nav__link" onClick={() => scrollTo(featuresRef)}>Features</button>
                        <button className="ds-nav__link" onClick={() => scrollTo(demoRef)}>Demo</button>
                        <button className="ds-nav__link" onClick={() => scrollTo(impactRef)}>Impact</button>
                        <button className="ds-nav__link" onClick={() => scrollTo(contactRef)}>Contact</button>
                        <button className="ds-btn ds-btn--primary ds-btn--sm" onClick={() => navigate('brain')}>
                            Launch App →
                        </button>
                    </div>

                    <button
                        className={`ds-hamburger ${menuOpen ? 'ds-hamburger--open' : ''}`}
                        onClick={() => setMenuOpen(o => !o)}
                        aria-label="Toggle menu"
                    >
                        <span /><span /><span />
                    </button>
                </div>
            </nav> */}

            {/* ── HERO ──────────────────────────────────────────────── */}
            <section className={`ds-hero ${animateHeader ? 'ds-hero--visible' : ''}`}>
                <div className="ds-hero__bg">
                    <div className="ds-orb ds-orb--1" />
                    <div className="ds-orb ds-orb--2" />
                    <div className="ds-orb ds-orb--3" />
                    <div className="ds-grid" />
                </div>
                <div className="ds-hero__content">
                    <span className="ds-badge">🔬 AI-Powered Healthcare Platform</span>
                    <h1 className="ds-hero__title">
                        <span className="ds-white">Drug</span><span className="ds-gradient">Seek</span>
                    </h1>
                    <p className="ds-hero__subtitle">
                        Next-generation AI for medical diagnostics,<br className="ds-br" />
                        drug discovery &amp; molecular analysis
                    </p>
                    <div className="ds-hero__actions">
                        <button className="ds-btn ds-btn--primary" onClick={() => openModule("lung")}>
                            Launch Platform
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                        </button>
                        <button className="ds-btn ds-btn--ghost" onClick={() => scrollTo(contactRef)}>
                            Contact Us
                        </button>
                    </div>
                </div>
                <div className="ds-hero__scroll-hint" onClick={() => scrollTo(featuresRef)}>
                    <div className="ds-scroll-dot" />
                    <span>Scroll to explore</span>
                </div>
            </section>

            {/* ── FEATURE CARDS ─────────────────────────────────────── */}
            <section
                className={`ds-features ${isVisible.features ? 'ds-features--visible' : ''}`}
                ref={featuresRef}
                id="features"
            >
                <div className="ds-section-head ds-section-head--center">
                    <span className="ds-label">Explore Modules</span>
                    <h2 className="ds-h2">Our AI-Powered Tools</h2>
                    <p className="ds-sub">Click any module to dive in — each powered by a specialized deep learning model</p>
                </div>

                <div className="ds-cards">
                    {features.map((f, i) => (
                        <button
                            key={f.id}
                            className="ds-card"
                            style={{ '--cc': f.color, '--cg': f.glow, '--cd': `${i * 70}ms` }}
                            onClick={() => openModule(f.id)}
                        >
                            <div className="ds-card__top">
                                <span className="ds-card__icon">{f.icon}</span>
                                <span className="ds-card__tag">{f.tag}</span>
                            </div>
                            <h3 className="ds-card__title">{f.title}</h3>
                            <p className="ds-card__desc">{f.desc}</p>
                            <div className="ds-card__footer">
                                <span className="ds-card__cta">
                                    Open Module
                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                                </span>
                            </div>
                            <div className="ds-card__shine" />
                        </button>
                    ))}
                </div>
            </section>

            {/* ── DEMO VIDEO ────────────────────────────────────────── */}
            <section
                className={`ds-demo ${isVisible.demo ? 'ds-demo--visible' : ''}`}
                ref={demoRef}
                id="demo"
            >
                <div className="ds-section-head ds-section-head--center">
                    <span className="ds-label">See It In Action</span>
                    <h2 className="ds-h2">Platform Demo</h2>
                    <p className="ds-sub">Watch all AI tools working — brain, lung, drug generation and more</p>
                </div>

                <div className="ds-demo__layout">
                    <div className="ds-demo__video-wrap">
                        <iframe
                            src="https://www.youtube.com/embed/u_GuJ1ZaIKU?rel=0"
                            title="DrugSeek Platform Demo"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                        />
                    </div>
                    <div className="ds-demo__links">
                        <p className="ds-demo__links-title">Additional Walkthroughs</p>
                        {[
                            { href: 'https://youtu.be/-8BvnisNSVM',   label: 'Project Explanation Video 1' },
                            { href: 'https://youtu.be/ShDlVK8YNJ0',   label: 'Project Explanation Video 2' },
                        ].map(v => (
                            <a key={v.href} href={v.href} target="_blank" rel="noopener noreferrer" className="ds-demo__yt-link">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="#ff0000"><path d="M21.6 7.2a2.7 2.7 0 00-1.9-1.9C18 5 12 5 12 5s-6 0-7.7.3A2.7 2.7 0 002.4 7.2 28 28 0 002 12a28 28 0 00.4 4.8 2.7 2.7 0 001.9 1.9C6 19 12 19 12 19s6 0 7.7-.3a2.7 2.7 0 001.9-1.9A28 28 0 0022 12a28 28 0 00-.4-4.8z"/><path fill="#fff" d="M10 15l5-3-5-3v6z"/></svg>
                                <span>{v.label}</span>
                            </a>
                        ))}
                        <a
                            href="https://github.com/eluriNarsimhulu/Drug_Seek-Gen_AI_Drug_Discovery"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="ds-demo__yt-link"
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.165 6.839 9.49.5.09.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.605-3.369-1.34-3.369-1.34-.455-1.157-1.11-1.465-1.11-1.465-.907-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.682-.103-.253-.446-1.27.098-2.646 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.376.203 2.393.1 2.646.64.698 1.028 1.59 1.028 2.682 0 3.841-2.337 4.687-4.565 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.741 0 .267.18.577.688.48C19.138 20.16 22 16.416 22 12c0-5.523-4.477-10-10-10z"/></svg>
                            <span>View on GitHub</span>
                        </a>
                    </div>
                </div>
            </section>

            {/* ── TECHNOLOGY ────────────────────────────────────────── */}
            <section
                className={`ds-tech ${isVisible.technology ? 'ds-tech--visible' : ''}`}
                ref={techRef}
                id="technology"
            >
                <div className="ds-section-head ds-section-head--center">
                    <span className="ds-label">Under the Hood</span>
                    <h2 className="ds-h2">Our Technological Edge</h2>
                </div>
                <div className="ds-tech__grid">
                    {[
                        { icon: '🤖', title: 'Generative AI',       desc: 'State-of-the-art transformer models for generating novel molecular structures and predicting medical outcomes.' },
                        { icon: '👁️', title: 'Computer Vision',     desc: 'Advanced CNN and ViT architectures that continuously improve diagnostic accuracy across medical imaging tasks.' },
                        { icon: '☁️', title: 'Cloud Infrastructure', desc: 'Robust HIPAA-compliant infrastructure enabling secure, scalable solutions for global healthcare collaboration.' },
                    ].map((t, i) => (
                        <div className="ds-tech__item" key={i} style={{ '--cd': `${i * 120}ms` }}>
                            <div className="ds-tech__icon">{t.icon}</div>
                            <h4>{t.title}</h4>
                            <p>{t.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── IMPACT ────────────────────────────────────────────── */}
            <section
                className={`ds-impact ${isVisible.impact ? 'ds-impact--visible' : ''}`}
                ref={impactRef}
                id="impact"
            >
                <div className="ds-section-head ds-section-head--center">
                    <span className="ds-label">By the Numbers</span>
                    <h2 className="ds-h2">Our Impact</h2>
                </div>
                <div className="ds-impact__grid">
                    {impacts.map((m, i) => (
                        <div className="ds-impact__card" key={i} style={{ '--cd': `${i * 100}ms` }}>
                            <span className="ds-impact__icon">{m.icon}</span>
                            <strong className="ds-impact__value">{m.value}</strong>
                            <span className="ds-impact__label">{m.label}</span>
                        </div>
                    ))}
                </div>
                <blockquote className="ds-testimonial">
                    <p>"DrugSeek's AI platform has revolutionized our research process, cutting development time in half."</p>
                    <cite>— Dr. Sarah Chen, Research Director</cite>
                </blockquote>
            </section>

            {/* ── CTA ───────────────────────────────────────────────── */}
            <section
                className={`ds-cta ${isVisible.cta ? 'ds-cta--visible' : ''}`}
                ref={ctaRef}
                id="cta"
            >
                <div className="ds-orb ds-orb--cta" />
                <h2>Ready to Transform Healthcare?</h2>
                <p>Join our community of innovators pushing the boundaries of medical technology and drug research.</p>
                <div className="ds-cta__actions">
                    <button className="ds-btn ds-btn--primary" onClick={() => navigate('brain')}>
                        Get Started
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                    <button className="ds-btn ds-btn--ghost" onClick={() => scrollTo(contactRef)}>Schedule a Demo</button>
                </div>
            </section>

            {/* ── CONTACT ───────────────────────────────────────────── */}
            <section className="ds-contact" ref={contactRef} id="contact">
                <div className="ds-section-head ds-section-head--center">
                    <span className="ds-label">Get in Touch</span>
                    <h2 className="ds-h2">Connect With Us</h2>
                    <p className="ds-sub">Have questions or want to explore collaboration? Our team is ready.</p>
                </div>
                <div className="ds-contact__grid">
                    <form className="ds-contact__form" onSubmit={e => e.preventDefault()}>
                        <div className="ds-field">
                            <label>Full Name</label>
                            <input type="text" placeholder="Your Name" />
                        </div>
                        <div className="ds-field">
                            <label>Email Address</label>
                            <input type="email" placeholder="your@email.com" />
                        </div>
                        <div className="ds-field">
                            <label>Subject</label>
                            <select>
                                <option value="">Select a subject</option>
                                <option>General Inquiry</option>
                                <option>Partnership Opportunity</option>
                                <option>Request Demo</option>
                                <option>Technical Support</option>
                            </select>
                        </div>
                        <div className="ds-field">
                            <label>Message</label>
                            <textarea rows="4" placeholder="Your message..."></textarea>
                        </div>
                        <button type="submit" className="ds-btn ds-btn--primary ds-btn--full">Send Message</button>
                    </form>

                    <div className="ds-contact__info">
                        {[
                            { icon: '📍', title: 'Location', text: 'Udaan Block-F, KMIT, Narayanaguda 500029, Telangana, India.' },
                            { icon: '📞', title: 'Phone',    text: '+91 9030180427' },
                            { icon: '✉️', title: 'Email',    text: 'drugseek.med@gmail.com' },
                        ].map((info, i) => (
                            <div className="ds-info-item" key={i}>
                                <span className="ds-info-icon">{info.icon}</span>
                                <div>
                                    <h4>{info.title}</h4>
                                    <p>{info.text}</p>
                                </div>
                            </div>
                        ))}
                        <div className="ds-social">
                            <h4>Follow Us</h4>
                            <div className="ds-social__links">
                                <a href="https://github.com/eluriNarsimhulu" target="_blank" rel="noopener noreferrer" className="ds-social__btn" aria-label="GitHub">G</a>
                                <a href="#" className="ds-social__btn" aria-label="LinkedIn">in</a>
                                <a href="#" className="ds-social__btn" aria-label="Twitter">𝕏</a>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── FOOTER ────────────────────────────────────────────── */}
            <footer className="ds-footer">
                <div className="ds-footer__inner">
                    <div className="ds-footer__brand">
                        <div className="ds-logo-chip">DS</div>
                        <div>
                            <strong>DrugSeek</strong>
                            <span>Revolutionizing Healthcare with AI</span>
                        </div>
                    </div>
                    <div className="ds-footer__links">
                        {[
                            { label: 'Tools',   items: ['Brain MRI', 'Lung CT', '3D Viewer', 'Protein Prediction', 'AutoDock'] },
                            { label: 'Company', items: ['About Us', 'Our Team', 'Careers', 'Contact'] },
                            { label: 'Legal',   items: ['Privacy Policy', 'Terms of Service', 'HIPAA Compliance'] },
                        ].map(col => (
                            <div key={col.label}>
                                <h5>{col.label}</h5>
                                {col.items.map(item => <a key={item} href="#">{item}</a>)}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="ds-footer__bottom">
                    <p>© 2025 DrugSeek AI. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
};

export default Home;
