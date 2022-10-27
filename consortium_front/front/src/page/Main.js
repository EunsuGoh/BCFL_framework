import axios from 'axios';
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import serverEndpoint from '../server_setting.json';

function Main() {
  const navigate = useNavigate();
  const location = useLocation();

  const createFLTask = ()=>{
    navigate('/CreateFLTask',{state:{
      tag: location.state.tag
    }})
  }
  const showRegisteredFLTasks = ()=>{
    navigate("/FLlist",{state:{
      tag:location.state.tag
    }})
  }

  const tag = location.state.tag
  return (
    <div className='Main' style={{
      display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"center"
    }}>
      {tag==='eval'?
      <Box sx={{
        display:"flex",
        flexDirection:"column",
        alignItems:"center",
        justifyContent:"center"
      }}>
        <Typography variant="h5" component="div">
        Welcome! Evaluator :)
        </Typography>
        <Button variant="text" sx={{
            width:"100%"
          }} onClick = {createFLTask} >Create FL Task</Button>
          <Button variant="text" sx={{
            width:"100%"
          }}>Show registered FL Tasks</Button>
      </Box> : <Box sx={{
        display:"flex",
        flexDirection:"column",
        alignItems:"center",
        justifyContent:"center"
      }}>
        <Typography variant="h5" component="div">
        Welcome! Trainer :)
        </Typography>
          <Button variant="text" sx={{
            width:"100%"
          }} onClick = {showRegisteredFLTasks}>Show registered FL Task</Button>
      </Box>
      }
     
    </div>
  );
}

export default Main;
