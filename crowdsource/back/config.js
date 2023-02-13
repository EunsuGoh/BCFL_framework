require('dotenv').config();
const { ACCESS_TOKEN_SECRET, HASHROUND } = process.env;

module.exports = {
    accessTokenSecret: ACCESS_TOKEN_SECRET,
    hashRound: HASHROUND,
};