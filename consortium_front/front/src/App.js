import { Route, Routes } from 'react-router-dom';
import Home from './page/Home';
import Main from './page/Main';
import FLresult from './page/FLresult';
import CreateFLTask from './page/CreateFLTask'
import ShowRegisteredFLTasks from './page/ShowRegisteredFLTasks'
import FLlist from './page/FLlist'
import Monitor from './page/Monitor'


function App() {
  return (
    <Routes>
      <Route exact path='/' element={<Home />} />
      <Route path='/Main' element={<Main />} />
      <Route path='/CreateFLTask' element={<CreateFLTask />} />
      <Route path='/ShowRegisteredFLTasks' element={<ShowRegisteredFLTasks />} />
      <Route path='/FLlist' element={<FLlist />} />
      <Route path='/Monitor' element={<Monitor />} />
    </Routes>
  );
}

export default App;
