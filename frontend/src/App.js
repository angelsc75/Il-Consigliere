import React from 'react';
import { Route, Routes } from 'react-router-dom';
import MovieRecommender from './MovieRecommender';
import OnboardingFlow from './OnboardingFlow';
import Header from './components/ui/header'; 
import './globals.css';

function App() {
  return (
    <>
      
      <Routes>
        <Route path="/" element={<OnboardingFlow />} />
        <Route path="/recommender" element={<MovieRecommender />} />
      </Routes>
    </>
  );
}

export default App;
