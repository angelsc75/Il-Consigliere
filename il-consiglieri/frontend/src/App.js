import MovieRecommender from './MovieRecommender';
import OnboardingFlow from './OnboardingFlow';

import logo from './logo.svg';
import './App.css';
<Route path="/onboarding" element={<OnboardingFlow />} />
function App() {
  return (
    <div className="App">
      <MovieRecommender />
    </div>
  );
}


export default App;
