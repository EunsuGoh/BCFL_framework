import axios from 'axios';
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import serverEndpoint from '../server_setting.json';
import TextField from '@mui/material/TextField';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';

function Monitor() {
  const navigate = useNavigate();
  const location = useLocation()
  const [clickedFlag, setClickedFlag] = useState("False")
  console.log(location.state.globalRounds)
  
  useEffect(()=>{
    console.log(clickedFlag)
  },[clickedFlag])

  const evalStartFL = async () =>{
    const setFlag = await axios.post(serverEndpoint.server+'/eval_setFlag')
    if(setFlag.data.msg){
      setClickedFlag("True")
      const result = await axios.post(serverEndpoint.server+'/eval_startEval')
      if(result.data.flag){
        window.alert(result.data.msg)
        setClickedFlag("False")
      }  else{
        console.log(result.data)
      }
    }else{
      window.alert("Train Falg setting failed")
    }
  }
  const trainCheckFlag = async () =>{
    const result = await axios.post(serverEndpoint.train+'/train_checkTrainFlag')
    if(result.data.msg === 'True'){
      window.alert("You can start training. Press PARTICIPATE please.")
    }else{
      window.alert("Evaluator doesn't start FL process yet, please wait")
    }
  }
  const trainStartFL = async () =>{
    const result = await axios.post(serverEndpoint.train+'/train_startTrain',{index:location.state.train_idx})
    if(result.data){
    }else{
    }
  }


  return (
    <div className='Monitor' style={{
      display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"center"
    }}>
      <Box sx = {{
          display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"center"
         }}>
      <Typography variant="h5" component="div">
        Monitoring Page
        </Typography>
        {location.state.tag === 'eval' ?
         <Box>
          <Typography variant="h6" component="div">
         You can start FL Task
         </Typography>
         <Button variant="text" sx={{
        width:"100%"
      }} onClick ={evalStartFL}>Start</Button>
         </Box> :
         <Box>
         <Typography variant="h6" component="div">
        1. Check Training Status
        </Typography>
        <Button variant="text" sx={{
       width:"100%"
     }} onClick = {trainCheckFlag}>Check</Button>
     <Typography variant="h6" component="div">
        2. Participate in Training
        </Typography>
        <Button variant="text" sx={{
       width:"100%"
     }} onClick={trainStartFL}>Participate</Button>
        </Box>}
      </Box>
      <Box></Box>
    </div>
  );
}

export default Monitor;
