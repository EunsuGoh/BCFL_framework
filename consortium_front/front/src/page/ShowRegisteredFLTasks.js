import axios from 'axios';
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import serverEndpoint from '../server_setting.json';

function ShowRegisteredFLTasks() {
  const navigate = useNavigate();
  const location = useLocation();

  const tag = location.state.tag
  return (
    <div className='Main' style={{
      display:"flex",
      flexDirection:"column",
      alignItems:"center",
      justifyContent:"cneter"
    }}>
      {tag==='eval'?
      <Box sx={{
        display:"flex",
        flexDirection:"column",
        alignItems:"center",
        justifyContent:"cneter"
      }}>
        <Typography variant="h5" component="div">
        Welcome! Evaluator :)
        </Typography>
        <Button variant="text" sx={{
            width:"100%"
          }}>Create FL Task</Button>
          <Button variant="text" sx={{
            width:"100%"
          }}>Show registered FL Tasks</Button>
      </Box> : <Box sx={{
        display:"flex",
        flexDirection:"column",
        alignItems:"center",
        justifyContent:"cneter"
      }}>
        <Typography variant="h5" component="div">
        Welcome! Trainer :)
        </Typography>
          <Button variant="text" sx={{
            width:"100%"
          }}>Show registered FL Tasks</Button>
      </Box>
      }
     
    </div>
  );
}

export default ShowRegisteredFLTasks;
