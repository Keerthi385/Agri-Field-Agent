import React from "react";
import { Link } from "react-router-dom";
import HeroSection from "../components/HeroSection";
import FeatureCards from "../components/FeatureCards";

export default function DashboardPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCards />
    </div>
  );
}

const cardStyle = {};