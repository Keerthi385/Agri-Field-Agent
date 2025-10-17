import React from "react";
import "./NatureTheme.css";

const features = [
  { icon: "🩺", title: "Crop Disease Detection", desc: "Upload a leaf photo to instantly identify diseases with AI.", link: "/disease" },
  { icon: "🌱", title: "Fertilizer Advice", desc: "Get precise fertilizer recommendations for your soil and crop.", link: "/fertilizer" },
  { icon: "🎙️", title: "Voice Q&A", desc: "Ask questions in your language and receive clear, spoken answers.", link: "/voice" },
  { icon: "⛅", title: "Weather Insights", desc: "Use your location to get real-time field context and suggestions.", link: "/context" },
];

export default function FeatureCards() {
  return (
    <section className="container section">
      <div className="section__header">
        <h2 className="section__title">Intelligent Tools for Every Farmer</h2>
        <p className="section__subtitle">Beautiful, fast, and focused on real outcomes in the field.</p>
      </div>

      <div className="grid grid--cards">
        {features.map((f, i) => (
          <a key={f.title} href={f.link} className="card glass hover-float fade-in-up" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="card__icon">{f.icon}</div>
            <h3 className="card__title">{f.title}</h3>
            <p className="card__desc">{f.desc}</p>
            <span className="card__cta">Open →</span>
          </a>
        ))}
      </div>
    </section>
  );
}


