
<p align="center">
  <img width="200" src="https://raw.githubusercontent.com/dykoffi/files/main/logoCIQLJSON2.png" alt="CIQL JSON Logo">
</p>

<h1 align="center" style="color:grey">CIQL JSON</h1>
<p style="font-size:18.5px; border-bottom:1px solid grey; padding-bottom:30px" align="justify">
    Ciql-JSON is a tool for manipulating JSON files. it allows to open and modify a JSON file and save results in another file or in the current file. You can also use it to create JSON files, it is very practical and easy to use.
</p>
<h1 style="color:#9fa8da;">Intallation</h1>

> yarn add ciql-json
> 
> npm i ciql-json

<h1 style="color:#9fa8da;">Usage</h1>


```js
const ciqlJSON = require('ciql-json')
```
<h3 id="funcopen" style="color:#ff80ab;">
<a href="#funcopen"># .open</a></h3>

<p style="font-size:16.5px">
you use the <code>open</code> method to open an existing JSON file. Once the file is linked you have access to the modification methods.
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .open("file.json")
    .set("address", {town : "", city : ""})
    .save()
```
<p style="font-size:16.5px">
<b>NB : </b> If you do not specify an output file in the <code>save</code> function, the change will be made in the input file.
</p>


<h3 id="funccreate" style="color:#ff80ab;">
<a href="#funccreate"># .create</a></h3>
<p style="font-size:16.5px">
Use the  <code>create</code>  methode to initialize a JSON object.
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({name:"Edy"})
    .set("address", {town : "Abidjan", city : "Marcory"})
    .save("me.json")

//output file (me.json): {name:"Edy", address : {town : "Abidjan", city : "Marcory"}
```

<p style="font-size:16.5px">
<b>NB : </b>In this case, If you do not specify an output file, there is an error because there, not input file.
</p>

<h3 id="funcsave" style="color:#ff80ab;">
<a href="#funcsave"># .save</a> </h3>
<p style="font-size:16.5px">
Use the <code>save</code> method to save the JSON data in current file or other file.
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({name : "Edy", school : "ESATIC"})
    .save("data.json")

//output file (data.json): {name:"Edy", school : "ESATIC"}
```

<h3 id="funcset" style="color:#ff80ab;">
<a href="#funcset">#  .set</a></h3>

<p style="font-size:16.5px">
Use the <code>set</code> method to add or modify the values ​​of the JSON object.
</p>


```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({})
    .set("school", "ESATIC")
    .set("location", "Treichville")
    .set("address", "")
    .save('data.json')

//output file (data.json): {school : "ESATIC", location : "Treichville", address : ""}
```

<p style="font-size:16.5px">
You can modify an object by specifying the key : 
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({school : {name : "ESATIC"}})
    .set("school.location", "Treichville")
    .save('data.json')

//output file (data.json): {school : {name : "ESATIC", location: "Treichville",}}
```

<h3 id="funcremove" style="color:#ff80ab;">
<a href="#funcremove">#  .remove</a></h3>

<p style="font-size:16.5px">
Use the <code>remove</code> method to delete key in JSON data.
</p>


```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({})
    .set("school", "ESATIC")
    .set("location", "Treichville")
    .remove("school")
    .save('data.json')

//output file (data.json): { location : "Treichville"}
```


<h3 id="funcextract" style="color:#ff80ab;">
<a href="#funcextract">#  .extract</a></h3>
<p style="font-size:16.5px">
Extract a schema in the JSON object following the destructuring of ES6.
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({name:"KOFFI", school : "ESATIC", address : "Abidjan"})
    .extract("{ school }")
    .save('data.json')

//output file (data.json): {school : "ESATIC"}
```


<h3 id="funcpush" style="color:#ff80ab;">
<a href="#funcpush">#  .pushTo</a></h3>
<p style="font-size:16.5px">
Add data to an array value.
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({school : "ESATIC", courses : []})
    .pushTo("courses","Data Sciences", "MERISE")
    .save('data.json')

//output file (data.json): {school : "ESATIC", courses : ["Data Sciences", "MERISE"]}
```

<h3 id="funcpop" style="color:#ff80ab;">
<a href="#funcpop">#  .popTo</a></h3>
<p style="font-size:16.5px">
Delete last data to an array value.
</p>

```js
const ciqlJson = require("ciql-json")

ciqlJson
    .create({school : "ESATIC", courses : []})
    .pushTo("courses","Data Sciences", "MERISE")
    .popTo("courses")
    .save('data.json')

//output file (data.json): {school : "ESATIC", courses : ["Data Sciences"]}
```

<h3 id="funcgetData" style="color:#ff80ab;">
<a href="#funcgetData"># .getData</a></h3>

<p style="font-size:16.5px">
Use <code>getData</code> if you want to return the final value of the json in a variable.
</p>


```js
const ciqlJson = require("ciql-json")

const data = ciqlJson
                .create({})
                .set("school", "ESATIC")
                .set("location", "Treichville")
                .getData()

//output : data = {school : "ESATIC", location : "Treichville"}
```
<h3 id="funcgetKeys" style="color:#ff80ab;">
<a href="#funcgetKeys"># .getKeys</a></h3>

<p style="font-size:16.5px">
<code>getKeys</code> return keys of the json Object.
</p>


```js
const ciqlJson = require("ciql-json")

const data = ciqlJson
                .create({})
                .set("school", "ESATIC")
                .set("location", "Treichville")
                .getKeys()

//output : data = ["school", "location" ]
```

<h3 id="funcgetValues" style="color:#ff80ab;">
<a href="#funcgetValues"># .getValues</a></h3>

<p style="font-size:16.5px">
<code>getValues</code> return values of the json Object.
</p>


```js
const ciqlJson = require("ciql-json")

const data = ciqlJson
                .create({})
                .set("school", "ESATIC")
                .set("location", "Treichville")
                .getValues()

//output : data = ["ESATIC", "Treichville" ]
```
<h1 style="color:#9fa8da;">Licence</h1>
<p>
MIT License

Copyright (c) 2021 dykoffi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</p>
<p align="center" style="font-size:12.5px">
LICENSE <code>MIT</code>
</p>