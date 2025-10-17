import React from "react";
import "./NatureTheme.css";

export default function HeroSection() {
  return (
    <section className="hero">
      <div className="hero__bg" />
      <div className="hero__glass fade-in-up">
        <h1 className="hero__title">
          Grow Smarter with <span className="hero__accent">AI</span>
        </h1>
        <p className="hero__subtitle">
          Diagnose crop diseases, get fertilizer advice, ask questions by voice,
          and track weather insights — all tailored to your field.
        </p>
        <div className="hero__ctaRow">
          <a href="/disease" className="btn btn--primary">Detect Disease</a>
          <a href="/market" className="btn btn--ghost">Market Prices</a>
        </div>
      </div>
    </section>
  );
}


