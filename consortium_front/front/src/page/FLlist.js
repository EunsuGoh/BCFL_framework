import axios from 'axios';
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import serverEndpoint from '../server_setting.json';

function FLlist() {
  const [model, setModel] = useState("");
  const [maxNumUpdates,setMaxNumUpdates] = useState("");
  const [globalRounds,setGlobalRounds] = useState("");
  const [trainers,setTrainers] = useState("");
  const location = useLocation();
  const navigate = useNavigate();

  const getTask = async () =>{
    const result = await axios.post(serverEndpoint.server+"/showFlTask")
    console.log(result.data.data['evaluator'])
    setModel(result.data.data['evaluator'].model)
    setMaxNumUpdates(result.data.data['evaluator'].maxNumUpdates)
    setGlobalRounds(result.data.data['evaluator'].globalRounds)
    setTrainers(result.data.data['evaluator'].trainers)
  }
  const startEval = async ()=>{
    navigate("/Monitor",{state:{
      tag: location.state.tag,
      
    }})
  }
  const startTrain = async ()=>{
    const result = await axios.post(serverEndpoint.train+"/train_participate")
    if(result.data.flag){
      window.alert(result.data.msg)
    }else{
      window.alert(result.data.msg)
      navigate("/Monitor",{state:{
        tag: location.state.tag,
        globalround:globalRounds,
        train_idx :result.data.train_idx
      }})
    }
    
  }
  useEffect(()=>{
    getTask()
  },[])
  useEffect(()=>{
    console.log(model,maxNumUpdates,globalRounds)
  },[model,maxNumUpdates,globalRounds])
  return (
    <div className='FLlist' style={{
      display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"cneter"
    }}>
     <Box sx={{
      display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"cneter"
     }}>
     <Typography variant="h5" component="div">
        Training Info...
        </Typography>
        {model.length>0 ?
        <div>
        <Typography variant="h6" component="div">
        model : {model}
        </Typography>
        <Typography variant="h6" component="div">
        global rounds : {globalRounds}
        </Typography>
        <Typography variant="h6" component="div">
        max number of updates : {maxNumUpdates}
        </Typography>
        <Typography variant="h6" component="div">
        trainers : {trainers}
        </Typography>
        </div> :
        <Typography variant="h6" component="div">
        No Data
        </Typography>}
        {location.state.tag ==='eval' ?  <Button variant="text" sx={{
            width:"100%"
          }} onClick = {startEval} >Evaluation Start</Button>
       :  <Button variant="text" sx={{
        width:"100%"
      }} onClick = {startTrain}>Participate</Button>}
     </Box>
    </div>
  );
}

export default FLlist;
